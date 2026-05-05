import os
import sys

__package__ = "trainer"  # 指定包名，用于相对导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    单个 epoch 的训练循环
    
    流程：
    1. 数据加载
    2. 前向传播 + 损失计算
    3. 梯度累积
    4. 梯度更新 + 学习率调度
    5. 日志记录 + 模型保存
    
    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器
        iters: 该 epoch 的总步数
        start_step: 起始步数（断点续训用）
        wandb: 日志记录工具（可选）
    """
    start_time = time.time()
    last_step = start_step
    
    # ===== 主训练循环 =====
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ===== 1. 数据移到设备 =====
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        
        # ===== 2. 学习率调度（Cosine Annealing） =====
        # 从初始 lr 逐步降低到 0，形成余弦衰减曲线
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ===== 3. 前向传播（混合精度） =====
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # 总损失 = LM 损失 + MoE 辅助损失
            loss = res.loss + res.aux_loss
            # 梯度累积：除以 accumulation_steps 使得累积后的梯度相当于更大 batch_size
            loss = loss / args.accumulation_steps

        # ===== 4. 反向传播（混合精度缩放） =====
        # GradScaler 自动处理梯度溢出
        scaler.scale(loss).backward()

        # ===== 5. 梯度更新 =====
        # 每累积 accumulation_steps 次梯度后执行一次优化器更新
        if step % args.accumulation_steps == 0:
            # 反缩放梯度（混合精度）
            scaler.unscale_(optimizer)
            
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 优化器步骤 + 缩放器更新
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度，准备下一个累积周期
            optimizer.zero_grad(set_to_none=True)

        # ===== 6. 日志记录 =====
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 恢复原始损失值
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60  # 预估剩余时间
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ===== 7. 定期保存模型 =====
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切换到 eval 模式（关闭 dropout）
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 获取原始模型（处理 DDP 和 torch.compile 的包装）
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            
            # 保存权重（转为 half 节省空间）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            # 保存完整检查点（用于恢复训练）
            lm_checkpoint(
                lm_config, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch, 
                step=step, 
                wandb=wandb, 
                save_dir='../checkpoints'
            )
            
            model.train()  # 切换回 train 模式
            del state_dict

        # ===== 8. 内存清理 =====
        del input_ids, labels, res, loss

    # ===== 最后一个 step 的补充处理 =====
    # 如果最后一个 step 还没满足 accumulation_steps，也需要更新
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ==================== 命令行参数 ====================
    parser = argparse.ArgumentParser(description="MiniMind 预训练脚本")
    
    # 保存参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="单卡 batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数（有效 batch = batch_size × accumulation_steps）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    
    # 设备和精度
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（bfloat16/float16）")
    
    # 数据加载
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    
    # 模型架构
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Transformer 层数")
    parser.add_argument('--max_seq_len', default=340, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE（0=否，1=是）")
    
    # 数据路径
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    
    # 权重和恢复
    parser.add_argument('--from_weight', default='none', type=str, help="从哪个权重继续训练（none=从零开始）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动恢复训练（0=否，1=是）")
    
    # 日志和可视化
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔（步数）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb 记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb 项目名")
    
    # 优化
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用 torch.compile 加速（0=否，1=是）")
    
    args = parser.parse_args()

    # ==================== 初始化 8 个步骤 ====================
    
    # ========== 1. 初始化环境和随机种子 ==========
    """
    分布式训练初始化：
    - 初始化进程组（多卡训练）
    - 设置随机种子确保可复现性
    """
    local_rank = init_distributed_mode()  # 返回本地 GPU 编号（单卡时为 0）
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 多卡时绑定到指定 GPU
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))  # 多卡时每个进程用不同的种子
    
    # ========== 2. 配置目录、模型参数、检查检查点 ==========
    """
    准备训练环境：
    - 创建输出目录
    - 初始化模型配置
    - 检查是否存在之前的检查点（用于续训）
    """
    os.makedirs(args.save_dir, exist_ok=True)  # 创建 out 目录
    
    # 创建模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    
    # 检查是否有检查点可以恢复
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    """
    混合精度训练：
    - 用 bfloat16/float16 计算前向传播和反向传播
    - 用 float32 保存权重
    - 节省 50% 的显存，加快计算
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # autocast_ctx：自动转换数据类型的上下文管理器
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 wandb/swanlab ==========
    """
    可视化训练过程：
    - 监控损失、学习率等指标
    - 支持断点续训时自动恢复原运行
    """
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 国内友好的替代品
        
        # 如果之前有运行，恢复原 run ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        
        # 构造运行名称
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    """
    核心训练组件初始化：
    - 模型：从零或从检查点加载
    - 数据加载器：支持分布式采样
    - 优化器：AdamW
    - 梯度缩放器：混合精度所需
    """
    # 初始化模型和 tokenizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 创建数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 分布式采样器（多卡时确保不重叠）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 梯度缩放器（混合精度所需）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从检查点恢复状态 ==========
    """
    断点续训：
    - 恢复模型权重
    - 恢复优化器状态（momentum、variance）
    - 恢复梯度缩放器状态
    - 恢复训练进度（epoch、step）
    """
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    """
    性能优化：
    - torch.compile：PyTorch 2.0 的编译加速（可选）
    - DistributedDataParallel：多卡训练
    """
    if args.use_compile == 1:
        model = torch.compile(model)  # 编译优化（第一次运行会比较慢）
        Logger('torch.compile enabled')
    
    # 分布式包装（如果是多卡）
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    """
    主训练循环：
    - 按 epoch 循环
    - 每个 epoch 中调用 train_epoch 执行一个完整的训练过程
    - 支持从指定 epoch 和 step 恢复
    """
    for epoch in range(start_epoch, args.epochs):
        # 设置 epoch 用于分布式采样（确保不同 epoch 打乱顺序不同）
        train_sampler and train_sampler.set_epoch(epoch)
        
        # 重新设置随机种子（保证 epoch 之间的随机性）
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        
        # 计算是否需要跳过前面的 step（续训时）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        
        # 创建批次采样器（支持跳过前面的批次）
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        
        # 创建数据加载器
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 执行本 epoch 的训练
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    """
    释放资源：
    - 销毁进程组
    - 释放通信资源
    """
    if dist.is_initialized():
        dist.destroy_process_group()