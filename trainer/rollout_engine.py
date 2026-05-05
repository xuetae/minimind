"""
如果使用 SGLang 加速，需先通过下面命令启动一个 transformers 格式模型服务：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998

这个文件负责“采样/rollout”阶段：
- 用本地 PyTorch 模型生成回复
- 或者通过 HTTP 调用 SGLang 服务生成回复
- 同时计算每个 token 的 logprob，供 RL / DPO / PPO 等流程使用
"""
import os
import sys

# 将项目根目录加入 import 路径，便于从 trainer 中导入 model 等模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """计算每个样本最后 n_keep 个 token 的对数概率。

    这个函数通常用于：
    - 计算生成结果的 token-level logprob
    - 在偏好学习/强化学习中构造奖励或 KL 相关项
    """
    if n_keep <= 0:
        # 如果不需要保留任何 token，则直接返回空张量
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    # DDP 包装模型需要先取出真实模型对象
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

    # 某些推理张量可能处于 inference 模式，克隆一份避免后续原地修改影响外部
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids

    # 只保留最后 n_keep + 1 个位置的 logits，前一个 token 用来预测后一个 token
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]

    per_token_logps = []
    # 逐样本取出最后 n_keep 个目标 token，并从对应 logits 中 gather 出它们的 logprob
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )

    # 堆叠成 [batch, n_keep]
    return torch.stack(per_token_logps)


@dataclass
class RolloutResult:
    """一次 rollout 的返回结果结构。

    记录完整输出、回复片段、token logprob、文本形式回复以及各种 mask / length 信息。
    """
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


class RolloutEngine(ABC):
    """rollout 引擎抽象基类：规定所有推理后端都要提供统一接口。"""
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """给定 prompt 生成若干条回复，并返回结构化结果。"""
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """当策略模型权重变化后，同步更新 rollout 引擎内部使用的权重来源。"""
        pass


class TorchRolloutEngine(RolloutEngine):
    """使用本地 PyTorch 模型进行 rollout 的实现。"""
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model  # 当前用于采样的策略模型
        self.tokenizer = tokenizer  # 用于 decode / special token id
        self.device = device  # 目标设备
        self.autocast_ctx = autocast_ctx  # 可选的 autocast 上下文，用于混合精度推理

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # DDP 时先拆出真实模型，避免重复包装带来的调用问题
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model

        # 如果外部没有提供 autocast 上下文，则使用空上下文
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            # 通过 generate 采样生成回复；repeat_interleave 用于一次为每个 prompt 生成 num_generations 条
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]

            # prompt 长度固定，后半段就是生成的 completion
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]

            # 组合成完整 mask，再计算每个生成 token 的 logprob
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            per_token_logps = compute_per_token_logps(
                self.policy_model,
                output_ids,
                completion_ids.size(1),
                attention_mask=full_mask,
            )

        # 将 token id 解码为文本，方便后续评价 / 日志记录
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # prompt_lens 和 completion_mask 的形状都与 batch 一致，便于后续 loss/score 对齐
        return RolloutResult(
            output_ids,
            completion_ids,
            per_token_logps,
            completions,
            prompt_ids.new_full((output_ids.size(0),), prompt_len),
            attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)),
        )

    def update_policy(self, model: torch.nn.Module):
        # 本地引擎只需要替换内部引用即可
        self.policy_model = model


class SGLangRolloutEngine(RolloutEngine):
    """通过 SGLang HTTP 服务进行 rollout 的实现。"""
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')  # 去掉末尾斜杠，避免拼接 URL 时出现重复 /
        self.shared_ckpt_path = shared_ckpt_path  # 用于与服务共享的临时权重目录
        self.timeout = timeout  # HTTP 请求超时时间
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # 与服务端保持一致的 tokenizer
        self.http = requests  # 保留 requests 模块句柄，便于统一调用

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # SGLang 通常需要去掉左侧 padding，所以先把有效 token 截出来
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()  # 只保留有效位
            input_ids_list.append(valid_ids)

        # 为每个 prompt 复制 num_generations 份，形成批量请求
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        # 请求体中包含输入、采样参数、以及是否返回 logprob
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }

        # 调用服务端 generate 接口
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()

        # 兼容单条/多条返回格式
        results = resp.json()
        if not isinstance(results, list):
            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []

        # 将返回结果整理成统一结构
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])

            # 服务端返回的 logprob 结构可能是嵌套列表，这里做一次兼容归一化
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)

            # 对齐长度：服务端若少返回则前面补 0，多返回则截断尾部
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []

            # 拼回完整输出，便于后续对齐 prompt 和 completion
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len

        def pad_to_tensor(seqs, max_len, pad_val=0):
            """把不同长度的列表补齐为定长 tensor。"""
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor(
                [[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids],
                device=device,
            ),
        )

    def update_policy(self, model: torch.nn.Module):
        # 该流程需要将最新权重保存到共享目录，并通知服务端重新加载
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                # DDP / compiled model 都要先拆开
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)

                # 将权重保存为 half + cpu，减少磁盘与传输开销
                abs_path = os.path.abspath(self.shared_ckpt_path)
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)

                # 通知 SGLang 服务从磁盘重新加载权重
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200:
                    print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}")
                ok = False

        # 多进程环境下，把结果广播给所有 rank，避免状态不一致
        if dist.is_initialized():
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0)
            dist.barrier()
            ok = bool(ok_t.item())

        if not ok:
            raise RuntimeError("SGLang update_policy failed")
        return ok

    def flush_cache(self) -> bool:
        """清空服务端缓存，通常在切换策略或长时间运行后使用。"""
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:
        """检查服务端健康状态。"""
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer=None,
    device: str = "cuda",
    autocast_ctx=None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    """工厂函数：根据 engine_type 构造对应的 rollout 引擎。"""
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
