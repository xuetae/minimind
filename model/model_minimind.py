import math, torch, torch.nn.functional as F  # math 用于数学运算；torch 是核心张量库；F 为函数式接口
from torch import nn  # 神经网络模块基类与层
from transformers.activations import ACT2FN  # 激活函数映射（如 gelu、silu）
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig  # HF 基类与生成混入
from transformers.modeling_outputs import MoeCausalLMOutputWithPast  # HF 输出类型，包含 past_kv

# ==================== MiniMind 配置类 ====================
class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型的配置类，继承自 transformers 的 PretrainedConfig
    用于定义模型的超参数（隐藏维度、层数、词表大小等）
    """
    model_type = "minimind"  # 模型类型标识
    
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # 基础架构参数
        self.hidden_size = hidden_size  # 隐藏层维度（Embedding 的输出维度）
        self.num_hidden_layers = num_hidden_layers  # Transformer 堆叠的层数
        self.use_moe = use_moe  # 是否启用 MoE 稀疏专家机制
        
        # 训练相关参数
        self.dropout = kwargs.get("dropout", 0.0)  # dropout 比例，用于训练时正则化
        
        # 词表和特殊 token
        self.vocab_size = kwargs.get("vocab_size", 6400)  # 模型的词表大小
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # BOS token id
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # EOS token id
        
        # 注意力机制参数
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否尝试使用 flash attention 接口
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # 查询头数（Q 头）
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # K/V 头数（可少于 Q 头以实现 GQA）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每头维度
        
        # FFN 参数
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # FFN 使用的激活函数名
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)  # FFN 中间层维度，按 64 对齐
        
        # 位置编码参数
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 最大支持的序列长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 的 eps 防止除零
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 基数，控制频率分布
        
        # 嵌入权重共享
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # 是否共享输入嵌入和输出 lm_head 权重
        
        # RoPE 长文本外推参数（YaRN）
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)  # 是否启用 RoPE 的 YaRN 外推
        self.rope_scaling = {
            "beta_fast": 32,  # 快速衰减 beta
            "beta_slow": 1,  # 慢速衰减 beta
            "factor": 16,  # 外推倍数
            "original_max_position_embeddings": 2048,  # 训练时的原始最大位置长度
            "attention_factor": 1.0,  # attention 缩放因子
            "type": "yarn"  # 表示使用 YaRN 方法
        } if self.inference_rope_scaling else None
        
        # MoE 相关参数（当 use_moe=False 时这些值会被忽略）
        self.num_experts = kwargs.get("num_experts", 4)  # 专家数量
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 选择的专家数 top-k
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # MoE 中间层大小
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)  # 是否归一化 top-k 权重
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # Router 辅助损失系数


# ==================== 核心算法模块 ====================
class RMSNorm(torch.nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）
    改进的 LayerNorm，只进行缩放，不进行中心化。
    相比 LayerNorm：
    - 计算更快（不需要计算均值）
    - 效果相当或更好
    - 内存开销更小
    
    公式：y = w * x / sqrt(mean(x²) + eps)
    RMS优势：相比于平均绝对值，其对极端值/大数值更敏感，更适合衡量“信号强度/特征幅度”，而不是中心化后的分布。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 数值稳定性系数
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放权重（与 LayerNorm 的 scale 相似），作用是调整每个特征维度的输出幅度，增强模型表达能力

    def norm(self, x):
        """计算 RMSNorm 的归一化部分"""
        # mean(x²) 在最后一维计算，保持前面维度
        # rsqrt 是倒数平方根：1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # element-wise 标准化

    def forward(self, x):
        """前向传播：对 x 进行 RMSNorm"""
        # 1. 先转为 float 避免数值精度问题
        # 2. 应用权重缩放
        # 3. 转回原数据类型（保持精度）
        return (self.weight * self.norm(x.float())).type_as(x)  # 保持输入的 dtype 输出相同 dtype


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算 RoPE（Rotary Position Embedding）的频率和旋转矩阵
    
    RoPE 的思想：
    - 为每个位置编码旋转角度
    - 不同频率的成分旋转速度不同
    - 支持任意序列长度的外推
    
    Args:
        dim: 位置编码的维度（通常等于 head_dim）
        end: 最大序列长度
        rope_base: 基数（通常为 1e6）
        rope_scaling: YaRN 长文本外推参数
    
    Returns:
        freqs_cos, freqs_sin: [seq_len, dim] 的余弦和正弦值，用于旋转 Q/K
    """
    # 计算基础频率：每个维度有不同的频率
    # 公式：θᵢ = rope_base^(-2i/d)，其中 i ∈ [0, d/2)
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0  # 每个偶数维度对应一个频率
    
    # YaRN：长文本外推方案
    # 当超过原始训练长度时，对低频进行缓冲，高频进行衰减
    if rope_scaling is not None:
        # 提取 YaRN 参数
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),  # 原始最大长度
            rope_scaling.get("factor", 16),  # 外推倍数
            rope_scaling.get("beta_fast", 32.0),  # 快速衰减的 beta
            rope_scaling.get("beta_slow", 1.0),  # 慢速衰减的 beta
            rope_scaling.get("attention_factor", 1.0)  # 注意力缩放因子
        )
        
        # 如果当前长度超过原始长度，应用 YaRN
        if end / orig_max > 1.0:  # 仅当目标长度超过训练长度时应用 YaRN 外推
            # 计算哪些维度应该被平滑插值
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            
            # 创建平滑的插值斜坡：0 到 1
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            
            # 应用插值：低频保留，高频衰减
            freqs = freqs * (1 - ramp + ramp / factor)
    
    # 为每个位置创建旋转矩阵
    t = torch.arange(end, device=freqs.device)  # 时间步索引 [0..end-1]
    freqs = torch.outer(t, freqs).float()  # 外积生成每个位置对应的相位角
    
    # 计算余弦和正弦值，并复制以覆盖完整维度
    # 因为 RoPE 通过旋转对复数表示的向量进行操作
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # 复制为完整维度
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin  # 返回 cos 与 sin 矩阵用于 RoPE


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    应用旋转位置编码到 Q/K 向量
    
    RoPE 的核心：通过旋转矩阵变换 Q/K，使得位置信息被编码到向量方向中
    
    Args:
        q: Query [batch, seq_len, num_heads, head_dim]
        k: Key [batch, seq_len, num_heads, head_dim]
        cos: 余弦值 [seq_len, dim]
        sin: 正弦值 [seq_len, dim]
    
    Returns:
        q_embed, k_embed: 应用了 RoPE 的 Q/K
    """
    def rotate_half(x):
        """将向量的后半段移动到前面并取负，等价于复数乘以 i 操作的一部分"""
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    
    # 将 cos/sin 扩展到与 q/k 的维度后做复数旋转
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed  # 返回经过 RoPE 的 q 与 k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 K/V 向量以匹配 Q 的头数
    
    用于 GQA（Grouped Query Attention）：
    - Q 有多个头（如 8）
    - K/V 头数更少（如 4）
    - 需要重复 K/V 以匹配 Q
    
    Args:
        x: [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 重复因子（num_q_heads // num_kv_heads）
    
    Returns:
        [batch, seq_len, num_q_heads, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape  # batch, seq_len, kv_heads, head_dim
    if n_rep == 1:
        return x  # 若无需重复，直接返回原始 K/V
    
    # 在新的维度上扩展并重排以实现重复
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


class Attention(nn.Module):
    """
    多头自注意力机制 + KV 缓存 + RoPE 位置编码 + Flash Attention
    
    核心流程：
    1. 将输入投影为 Q/K/V
    2. 应用 RoPE 位置编码
    3. 处理 KV 缓存（推理时加速）
    4. 计算注意力权重（support Flash Attention）
    5. 投影回原维度
    
    GQA（Grouped Query Attention）优化：
    - Q 有更多头，K/V 有较少头
    - 减少内存占用和计算量
    - 保持表达能力
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 计算多头注意力的参数
        self.num_key_value_heads = (config.num_attention_heads if config.num_key_value_heads is None 
                                    else config.num_key_value_heads)  # K/V 头数优先级
        self.n_local_heads = config.num_attention_heads  # Q 的头数
        self.n_local_kv_heads = self.num_key_value_heads  # K/V 实际头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 重复因子用于 GQA
        self.head_dim = config.head_dim  # 每头维度
        self.is_causal = True  # 因果掩码标识（只看过去）
        
        # 投影层：将 hidden_size 投影为多头的拼接表示
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        
        # 输出投影：将多头拼接后的向量投回 hidden_size
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        # 对 Q/K 使用 RMSNorm 以稳定训练
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # Dropout 层：用于 attention 权重与残差输出
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        
        # 检查是否可用 FlashAttention 接口并且用户启用了该选项
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        前向传播
        
        Args:
            x: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) RoPE 编码
            past_key_value: 之前步骤的 (K, V) 缓存，用于推理加速
            use_cache: 是否返回 K/V 缓存
            attention_mask: 注意力掩码
        
        Returns:
            output: [batch, seq_len, hidden_size]
            past_kv: 新的 (K, V) 缓存
        """
        bsz, seq_len, _ = x.shape  # batch size, sequence length, hidden_size
        
        # ===== 第 1 步：投影到 Q/K/V =====
        # 线性投影得到平铺的 Q/K/V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # reshape 为多头表示：batch, seq, num_heads, head_dim
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        # ===== 第 2 步：应用 RMSNorm 和 RoPE =====
        # 对 Q/K 做 RMSNorm，再加上 RoPE 位置信息
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings  # RoPE 的 cos 与 sin 矩阵
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        
        # ===== 第 3 步：处理 KV 缓存（推理加速）=====
        # 如果有历史的 K/V 缓存（推理模式），将其拼接以避免重复计算
        if past_key_value is not None:
            # 拼接之前缓存的 K/V 和当前的 K/V
            # 这样可以避免每步重新计算所有 token 的 attention
            xk = torch.cat([past_key_value[0], xk], dim=1)  # concat 在 seq 维上
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        # 如果需要缓存本次的 K/V，返回给上层用于下一步推理
        past_kv = (xk, xv) if use_cache else None
        
        # ===== 第 4 步：GQA 处理（复制 K/V 以匹配 Q 头数）=====
        # 转置为 attention 接受的形状：batch, num_heads, seq_len, head_dim
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # 对 K 重复以匹配 Q 的头数
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # 对 V 重复
        
        # ===== 第 5 步：计算注意力 =====
        # 尝试使用高效的 Flash Attention 实现（在满足若干条件下）
        if (self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))):
            # PyTorch 提供的 scaled_dot_product_attention 接口
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal
            )
        else:
            # 标准注意力计算
            # score[i,j] = Q[i] · K[j]^T / sqrt(d)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [b,h,seq,seq]
            
            # 应用因果掩码（下三角）：未来 token 不能被看到
            if self.is_causal:
                # 对上三角（未来位置）填充 -inf，防止模型看到未来信息
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            
            # 如果存在 attention_mask（如 padding），将对应位置的 score 置为非常小
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            
            # softmax -> dropout -> 与 V 加权求和
            output = self.attn_dropout(
                F.softmax(scores.float(), dim=-1).type_as(xq)
            ) @ xv
        
        # ===== 第 6 步：投影回原维度 =====
        # 将多头输出拼回并做输出投影 + 残差 dropout
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        
        return output, past_kv  # 返回输出向量与可选的 K/V 缓存

class FeedForward(nn.Module):
    """
    前馈网络（Feed Forward Network）
    
    使用 SwiGLU 激活函数的 FFN：
    - 更好的性能（相比 ReLU、GELU）
    - 参数量：hidden_size × intermediate_size × 2
    
    结构：
    hidden -> gate_proj -> SiLU -> * (element-wise)
                                   -> down_proj -> output
    hidden -> up_proj ---------> /
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        
        # SwiGLU 的三个线性层：gate_proj（门分支），up_proj（值分支），down_proj（投回 hidden）
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 门控分支
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # 投影回 hidden_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)    # 值分支
        
        # 获取激活函数（如 siLU）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """SwiGLU 前向：gate_proj->act * up_proj -> down_proj 返回原始维度"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """
    混合专家（Mixture of Experts）前馈网络
    
    原理：
    1. Router 决定每个 token 由哪些专家处理
    2. 选择 top-k 概率最高的专家
    3. token 在这些专家上的输出加权求和
    4. 只有被选中的专家参与计算（稀疏激活）
    
    优势：
    - 参数效率：同等训练参数下，激活参数更多
    - 模型容量：4 个 26M 专家 > 1 个 104M 模型
    - 但训练更复杂（需要平衡专家负载）
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        
        # Router：将输入映射到每个专家的得分（logit）
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # 创建多个专家，每个专家都是一个独立的 FeedForward
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数（可能在辅助操作中使用）

    def forward(self, x):
        """
        MoE 前向传播
        
        流程：
        1. Router 计算每个 token 对每个专家的相似度
        2. 选择 top-k 专家
        3. 计算加权输出
        4. 计算辅助损失（防止负载不均衡）
        """
        batch_size, seq_len, hidden_dim = x.shape  # 记录原始形状
        x_flat = x.view(-1, hidden_dim)  # 展平为 [batch*seq, hidden]
        
        # ===== 第 1 步：Router 计算专家权重 =====
        # Router logits -> softmax 得到每个 token 对每个专家的概率分布
        scores = F.softmax(self.gate(x_flat), dim=-1)  # [B*S, num_experts]
        
        # ===== 第 2 步：选择 top-k 专家 =====
        # 选取 top-k 专家：返回权重与对应专家索引
        topk_weight, topk_idx = torch.topk(
            scores,
            k=self.config.num_experts_per_tok,
            dim=-1,
            sorted=False
        )
        
        # 对 top-k 权重做归一化（避免数值不稳定）
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        
        # ===== 第 3 步：分发到专家并加权求和 =====
        # 为所有 token 初始化输出容器
        y = torch.zeros_like(x_flat)
        # 遍历每个专家，收集分配给该专家的 token 并计算其输出
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)  # 标记哪些 top-k 里含有当前专家 i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 这些 token 的全局索引
                weight = topk_weight[mask].view(-1, 1)  # 对应的权重
                # 计算专家输出并按权重累加到 y 对应索引
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 在训练时执行一个零乘操作以保证专家参数会收到梯度
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        
        # ===== 第 4 步：计算 Router 的辅助损失以鼓励专家负载均衡（仅在训练时） =====
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # 每个专家的负载
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        
        return y.view(batch_size, seq_len, hidden_dim)  # 恢复原始形状返回


class MiniMindBlock(nn.Module):
    """
    单个 Transformer 块
    
    结构（Pre-Norm）：
    input -> norm -> attention -> add residual -> norm -> FFN -> add residual -> output
    
    Pre-Norm 的优势：
    - 梯度流更稳定
    - 较深的模型也能训练
    - 不需要 warm-up learning rate
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)  # 自注意力子层
        
        # Pre-Norm：在 attention/ffn 前分别做 RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 根据配置选择普通 FFN 或 MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        前向传播
        
        流程：
        1. 注意力 + 残差连接
        2. FFN + 残差连接
        """
        # Attention 分支：Pre-Norm -> Attention -> 残差连接
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        
        # FFN 分支：Pre-Norm -> MLP -> 残差连接
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 的核心 Transformer 模型
    
    构成：
    - Token 嵌入层
    - 8 层 MiniMindBlock
    - 最后的 Layer Norm
    - 预计算的 RoPE 位置编码
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        
        # ===== 嵌入层 =====
        # Token -> 嵌入 -> dropout
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)  # 输入嵌入上的 dropout
        
        # ===== Transformer 层堆叠 =====
        # ModuleList 保持层有序
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config)
            for l in range(self.num_hidden_layers)
        ])
        
        # 输出前的 RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 预计算 RoPE 所需的 cos/sin 矩阵以便快速查表
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        # 注册为 buffer（不参与梯度更新，但会随模型保存/加载）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """
        前向传播
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]（1 表示有效，0 表示 padding）
            past_key_values: 之前步骤的 KV 缓存
            use_cache: 是否返回 KV 缓存
        
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            presents: 新的 KV 缓存
            aux_loss: MoE 辅助损失
        """
        batch_size, seq_length = input_ids.shape  # 获取 batch 与序列长度
        
        # 兼容不同格式的 past_key_values，如果不合理则置空
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算当前片段在整个序列中的起始位置（用于 RoPE 索引）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # Token -> 嵌入 -> dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # [B, S, H]
        
        # 在某些 runtime（如 meta-device）中 buffer 可能被重置，做一次恢复检查
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)
        
        # 切片得到当前序列长度对应的 RoPE 编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        
        # 逐层前向传播并收集每层的 present_key_value（用于缓存）
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        
        # 最后做一次 RMSNorm 归一化
        hidden_states = self.norm(hidden_states)
        
        # 汇总每层 MoE 的辅助损失（如果有的话），否则返回 0
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )
        
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    用于因果语言建模的 MiniMind 模型
    
    架构：
    - MiniMindModel（Transformer 主体）
    - LM Head（投影到词表）
    
    功能：
    - 训练：计算下一个 token 的预测损失
    - 推理：自回归生成文本
    """
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 权重共享
    
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # Transformer 主体
        self.model = MiniMindModel(self.config)
        
        # 语言建模头：[hidden_size] -> [vocab_size]
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # 权重共享（嵌入层和输出层共享权重，减少参数）
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, 
                logits_to_keep=0, labels=None, **kwargs):
        """
        前向传播
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            past_key_values: KV 缓存（推理时用）
            use_cache: 是否返回 KV 缓存
            labels: [batch, seq_len]（如果提供，计算损失）
            logits_to_keep: 只保留最后 N 个 logits（推理加速）
        
        Returns:
            MoeCausalLMOutputWithPast 对象，包含：
            - loss: 训练损失
            - logits: 预测的 logits
            - past_key_values: KV 缓存
        """
        # 将输入通过 Transformer 主体得到隐藏状态与缓存
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        
        # 可选地只保留最后 N 个 logits 用于推理加速
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # [batch, seq, vocab]
        
        # 计算损失（训练时）
        loss = None
        if labels is not None:
            # 对因果 LM 而言，预测位置 i 对应标签位置 i+1
            x = logits[..., :-1, :].contiguous()  # 去掉最后一个 time-step 的 logits
            y = labels[..., 1:].contiguous()      # 去掉第一个 label，因为它没有被预测
            
            # 交叉熵损失，忽略标注为 -100 的位置（通常为 padding）
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)),
                y.view(-1),
                ignore_index=-100
            )
        
        # 返回 HF 兼容的输出对象，包含 loss, aux_loss, logits, past_key_values 等
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
    
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, 
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, 
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """
        自回归生成文本
        
        流程：
        1. 重复输入以生成多个序列（num_return_sequences）
        2. 循环生成 max_new_tokens 个 token
        3. 每步：计算 logits -> 采样 -> 追加到序列
        4. 直到生成 EOS 或达到最大长度
        
        Args:
            inputs: [batch, seq_len] 的输入 token IDs
            max_new_tokens: 最多生成多少个新 token
            temperature: 采样温度（>1 更随机，<1 更确定）
            top_p: nucleus sampling 的概率阈值
            top_k: top-k sampling 的 k 值
            do_sample: True=采样，False=贪心解码
            repetition_penalty: 对已生成 token 的惩罚（避免重复）
        
        Returns:
            [batch, seq_len + max_new_tokens] 的生成 token IDs
        """
        # 支持多返回序列：将输入重复 num_return_sequences 次
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)  # 跟踪哪些序列已完成
        
        if streamer:
            streamer.put(input_ids.cpu())  # 如果有 streamer，则先发送初始输入
        
        # 迭代生成每一个新 token
        for _ in range(max_new_tokens):
            # 计算已经缓存的 past 长度，用于只前向新生成 token
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            
            # 仅把新增的后缀 token 作为输入（用于 caching）
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values,
                use_cache=use_cache, **kwargs
            )
            
            # 每步生成一个 token 时，attention_mask 需要扩展一个 1
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones(attention_mask.shape[0], 1)
                ], -1)
            
            # 取最后一个位置的 logits 并进行温度缩放
            logits = outputs.logits[:, -1, :] / temperature
            
            # 对已生成 token 应用重复惩罚
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            
            # Top-K 过滤：保留 top_k 个 logit，其它置 -inf
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            
            # Top-P (nucleus) 采样过滤：保留累计概率小于 top_p 的 logits
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            
            # 根据 do_sample 决定是采样还是贪心选择
            next_token = (torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                         if do_sample else torch.argmax(logits, dim=-1, keepdim=True))
            
            # 对已经完成的序列填充 EOS，避免后续改变
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            
            # 将新 token 追加到输入序列末尾并更新缓存
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            
            if streamer:
                streamer.put(next_token.cpu())  # 将新 token 发到 streamer
            
            # 标记完成的序列并在全部完成时退出生成循环
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break
        
        if streamer:
            streamer.end()  # 结束 streamer 输出
        
        # 根据参数返回 KV 或仅返回生成的 ids
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
