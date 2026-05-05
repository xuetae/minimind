import torch
from torch import optim, nn


# LoRA（Low-Rank Adaptation）简易实现
# 目标：在不修改原始权重的大前提下，通过低秩矩阵对权重更新进行参数高效的微调
class LoRA(nn.Module):
    # 构造函数：接受输入/输出维度与低秩秩值
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA 的秩 r，控制低秩近似的大小
        # A: 从输入到低秩空间的线性变换（不带偏置）
        self.A = nn.Linear(in_features, rank, bias=False)
        # B: 从低秩空间到输出的线性变换（不带偏置）
        self.B = nn.Linear(rank, out_features, bias=False)
        # A 使用小方差高斯初始化，便于训练稳定
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B 初始化为 0，使得初始化时 LoRA 对原模型没有影响
        self.B.weight.data.zero_()

    # 前向：先通过 A 降维，再通过 B 恢复到输出维度
    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """
    在模型中为满足条件的线性层动态绑定 LoRA 模块：
    - 仅对方形权重矩阵的线性层进行注入（通常为自注意力中的投影矩阵）
    - 将原来的 forward 替换为 原始输出 + LoRA 输出
    """
    for name, module in model.named_modules():
        # 仅处理 nn.Linear 且权重为方阵的层（常见于 Q/K/V/O 投影）
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 在原模型所在设备上构建 LoRA 子模块
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(next(model.parameters()).device)
            # 把 lora 作为属性绑定到该层，便于保存/加载
            setattr(module, "lora", lora)
            # 保存原始 forward 以便在新 forward 中复用
            original_forward = module.forward

            # 定义新的 forward：原始输出 + lora 输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 覆盖该线性层的 forward
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    从磁盘加载 LoRA 权重并注入到模型中；
    - 支持保存时带 module. 前缀的 checkpoint（DataParallel 保存格式）
    """
    # 加载权重到指定设备（跟模型一致）
    state_dict = torch.load(path, map_location=next(model.parameters()).device)
    # 兼容 module. 前缀（多卡保存时可能存在）
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 遍历模型模块，把对应的 lora 参数加载进去
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从 state_dict 筛选出属于当前模块 lora 的键，并去掉前缀
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 加载局部状态字典
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    仅保存模型中的 LoRA 参数到磁盘，以便后续合并或共享。
    - 将参数转换为 half 减少磁盘占用
    - 支持 DataParallel 的 module. 前缀清理
    """
    # 获取原始模型（如果模型被包装过，例如 DDP 或 _orig_mod）
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    # 遍历模块，收集有 lora 的模块参数
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 清理 DataParallel 前缀
            clean_name = name[7:] if name.startswith("module.") else name
            # 将 lora 的参数按命名空间保存（例如 bert.encoder.layer.0.attn.lora.A.weight）
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    # 写盘
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """
    将 LoRA 参数合并到原始权重并保存合并后的完整权重：
    - 读取 LoRA 参数并注入
    - 对每个线性层，计算 W + B*A 并写入 state_dict
    """
    # 先把 lora 参数加载到模型
    load_lora(model, lora_path)
    # 获取原始模型对象
    raw_model = getattr(model, '_orig_mod', model)
    # 拷贝原始参数（排除 lora 自己的命名）并转为 half 节省空间
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    # 遍历线性层，将 lora 的低秩增量叠加到权重上
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 确保保存当前权重（clone 避免原地修改）
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            # 如果该层存在 lora，则将 B @ A 加到权重上
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    # 保存合并后的权重到 save_path
    torch.save(state_dict, save_path)
