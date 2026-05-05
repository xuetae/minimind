"""模型格式转换脚本。

这个文件负责在 MiniMind 原生 PyTorch 权重、Transformers 权重、以及 LoRA 合并结果之间互相转换。
"""

import os
import sys
import json

# 把项目根目录加入搜索路径，便于从 scripts 目录直接运行。
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora

# 忽略用户级 warning，避免转换过程输出过多噪音。
warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    """把 MiniMind 原生权重转换成 MiniMind 的 Transformers 兼容格式。"""
    # 注册自动类，方便 save_pretrained / from_pretrained 使用。
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    # 用当前全局配置构建原生模型结构，再把 PyTorch 权重灌进去。
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)

    # 转成目标精度，通常是 float16 以节省显存和磁盘。
    lm_model = lm_model.to(dtype)
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')

    # 保存成 Transformers 目录结构。
    lm_model.save_pretrained(transformers_path, safe_serialization=False)

    # 复制 tokenizer 到目标目录，确保模型和词表配套。
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # 兼容 transformers 5.x 对 tokenizer/config 的额外要求。
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        config_path = os.path.join(transformers_path, "config.json")

        # tokenizer_config 中补上 tokenizer_class 和空的 extra_special_tokens。
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        tokenizer_config = {**tokenizer_config, "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        # config.json 中修正 rope 参数字段，保持和当前模型实现一致。
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['rope_theta'] = lm_config.rope_theta
        config['rope_scaling'] = None
        del config['rope_parameters']
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    """把 MiniMind 原生权重转换成 Qwen3/Llama 风格的 Transformers 格式。"""
    # 加载原始 PyTorch 权重。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)

    # 把 MiniMind 配置映射到 Qwen3 结构需要的通用字段。
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": lm_config.tie_word_embeddings
    }

    # 根据是否启用 MoE，选择普通 Qwen3 或 Qwen3 MoE 配置。
    if not lm_config.use_moe:
        qwen_config = Qwen3Config(
            **common_config,
            use_sliding_window=False,
            sliding_window=None
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)

        # transformers 5.x 对 MoE 权重命名和形状做了变化，这里做兼容重排。
        if int(transformers.__version__.split('.')[0]) >= 5:
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                new_sd[f'{p}.gate_up_proj'] = torch.cat([
                    torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]),
                    torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])
                ], dim=1)
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    # 严格加载权重，确保转换前后结构一一对应。
    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype)
    qwen_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')

    # 复制 tokenizer，保证新目录可以直接被 AutoTokenizer 识别。
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # 兼容 transformers 5.x 对 tokenizer_config 和 config 的字段要求。
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        config_path = os.path.join(transformers_path, "config.json")

        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        tokenizer_config = {**tokenizer_config, "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['rope_theta'] = lm_config.rope_theta
        config['rope_scaling'] = None
        del config['rope_parameters']
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    """把 Transformers 格式模型重新导出回原生 PyTorch 权重。"""
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    """把 base 权重和 LoRA 权重合并，导出成单独的 PyTorch checkpoint。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    apply_lora(lm_model)
    merge_lora(lm_model, lora_path, merged_torch_path)
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")


def convert_jinja_to_json(jinja_path):
    """把 Jinja chat template 文件转成 JSON 字符串输出，方便写入 config。"""
    with open(jinja_path, 'r') as f:
        template = f.read()
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')


def convert_json_to_jinja(json_file_path, output_path):
    """从 tokenizer_config.json 中提取 chat_template 并写成单独的 jinja 文件。"""
    with open(json_file_path, 'r') as f:
        config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f:
        f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


if __name__ == '__main__':
    # 这里给出一个默认配置，方便直接运行脚本做格式转换。
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # 默认把原生 torch 权重转成 transformers 目录。
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3'
    convert_torch2transformers(torch_path, transformers_path)

    # 下面这些转换示例默认注释掉，用户可按需打开。
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
