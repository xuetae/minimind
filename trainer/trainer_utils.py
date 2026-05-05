"""
trainer 工具集合：
- 提供分布式初始化、随机种子设置、检查点保存/加载、模型初始化等实用函数
- 这些函数在多个训练脚本中被复用
"""
import os
import sys

# 将当前包目录上级添加到 sys.path，方便相对导入 workspace 中的模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel
from model.model_minimind import MiniMindForCausalLM


def get_model_params(model, config):
    """计算并打印模型参数规模（以 M 为单位），对 MoE 做特殊处理以展示激活参数规模。

    说明：如果模型包含专家（experts），会计算 base/active 两种计数方式并以 `Total` 或 `Total-Aactive` 格式打印。
    """
    # 模型总参数数（百万）
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # MoE 相关的配置项（兼容不同字段名）
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)

    # 计算单个专家的参数量（寻找命名空间中的专家子模块）
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6

    # base: 去除所有专家参数后的基线参数量
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    # active: 考虑每个 token 实际激活的专家数后的近似激活参数量
    active = base + (expert * n_active) + (shared_expert * n_shared)

    # 如果 active 小于 total，打印两者对比（表示稀疏激活节省的参数）
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """判断当前进程是否为主进程（rank 0），用于控制日志打印与单点保存等操作。"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """只在主进程打印日志，避免多卡重复输出。"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """基于余弦退火的简单学习率计算器（缩放在 0.1-0.55 范围内）。

    返回值：float 学习率
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """初始化分布式训练环境（基于环境变量 RANK/LOCAL_RANK）。

    如果未设置 RANK（单机单卡或单进程模式），返回 0 表示非 DDP 模式。
    否则使用 NCCL 后端并绑定本地 GPU。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        # 未设置分布式环境，使用单卡
        return 0

    # 初始化进程组（NCCL），并返回 local rank
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """统一设置随机种子，保证可复现性（尽可能）。

    包括 Python random、numpy、torch 以及 CUDA 的随机数种子，并关闭 cudnn 的非确定性优化。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """保存或加载训练检查点。

    - 当传入 `model` 时：执行保存操作，写出两个文件：权重文件(.pth) 与 resume 文件(_resume.pth)
    - 当 `model` 为 None 时：尝试加载 resume 文件并返回其内容（用于断点续训）

    `kwargs` 支持额外的状态对象（如 lr_scheduler、scaler 等），会一并保存。
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ==== 保存模式 ====
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # 将权重转为 half 并移动到 cpu，以减小保存体积
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 原子性写入：先写 tmp 文件，再替换
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 收集 wandb run id（兼容不同 wandb API）
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 组装 resume 数据结构，包含模型、优化器、epoch、step、world_size 以及其他额外状态
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }

        # 将额外的对象也写入 resume（支持 DDP 包装对象的解包）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 释放显存和临时对象
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        # ==== 加载模式（仅返回 resume 数据）====
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            # 兼容不同 world_size 的恢复：按比例调整 step
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """初始化 tokenizer 与模型，并可选择加载已有权重。

    返回： (model.to(device), tokenizer)
    """
    # tokenizer 用于数据编码/解码
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # 构建模型实例
    model = MiniMindForCausalLM(lm_config)

    # 如果指定了已有权重（非 'none'），则尝试加载
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # 打印模型参数信息并返回
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """Batch sampler 支持跳过前若干个 batch（用于断点续训）

    用法：传入原始 sampler（或 index 列表）、batch_size 与需要跳过的 batch 数量。
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                # 如果还没跳过足够的 batches，则丢弃当前 batch
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 如果尾部还有不满 batch 的样本，且跳过计数满足条件，则产出该 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        # 估算总批次数（向上取整），并减去需要跳过的批次数
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """简易的奖励模型封装：用于在 RLHF 流程中对生成文本进行评分。"""
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        # 使用 AutoTokenizer/AutoModel 加载支持自定义实现的远程模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """根据对话历史与候选回复计算打分，返回限制在 [-3, 3] 的实数。"""
        # 将对话历史合并为一句上下文文本
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        # 假设底层模型实现了 get_score(tokenizer, messages) 的接口
        score = self.model.get_score(self.tokenizer, eval_messages)
        # 限幅返回，避免极端值
        return max(min(score, 3.0), -3.0)