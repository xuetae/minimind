"""
model 包初始化文件。

这里将常用的模型类与 LoRA 工具函数导出，方便从外部直接导入：
from model import MiniMindModel, MiniMindConfig, MiniMindForCausalLM, LoRA, apply_lora, load_lora, save_lora, merge_lora

本文件仅作导出与简单说明，不包含复杂逻辑。
"""

# 将模型及 LoRA 接口暴露在包级别，方便外部直接导入使用
from .model_minimind import MiniMindConfig, MiniMindModel, MiniMindForCausalLM
from .model_lora import LoRA, apply_lora, load_lora, save_lora, merge_lora

# 指定包的公共导出名称
__all__ = [
	"MiniMindConfig",
	"MiniMindModel",
	"MiniMindForCausalLM",
	"LoRA",
	"apply_lora",
	"load_lora",
	"save_lora",
	"merge_lora",
]
