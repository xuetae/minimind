from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizer 多进程警告

# ==================== 数据预处理函数 ====================

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    预处理聊天对话数据
    
    功能：
    1. 保留工具调用数据（含 'tools' 字段）
    2. 概率性地为普通对话添加系统提示（system prompt）
       - 增加多样性
       - 教会模型理解系统指令
    
    Args:
        conversations: [{"role": "user"/"assistant"/"system", "content": "..."}, ...]
        add_system_ratio: 添加 system 的概率
    
    Returns:
        预处理后的 conversations
    """
    # 工具调用数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations):
        return conversations

    # 系统提示的候选列表（中英文混合）
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    
    # 以 add_system_ratio 的概率添加 system 消息
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """
    后处理聊天文本
    
    功能：
    - 概率性地移除空的 <think></think> 标签
    - 减少模型的无意义思考
    
    Args:
        prompt_content: 格式化后的对话文本
        empty_think_ratio: 保留空思考标签的概率（默认 0.2 表示 80% 概率移除）
    
    Returns:
        处理后的文本
    """
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


# ==================== 数据集类 ====================

class PretrainDataset(Dataset):
    """
    预训练数据集
    
    格式：JSONL，每行一条数据
    例子：{"text": "这是一段预训练文本"}
    
    处理流程：
    1. Tokenize：文本 -> token ID 序列
    2. 添加 BOS/EOS 标记
    3. Padding：补齐到 max_length
    4. 掩码构造：标记有效部分（非 padding）
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 库加载 JSONL
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取单个样本
        
        返回：(input_ids, labels)
        - 两者形状都是 [max_length]
        - labels 中 padding 部分为 -100（损失函数会忽略）
        """
        sample = self.samples[index]
        
        # Tokenize：文本 -> token ID
        # add_special_tokens=False 是因为我们手动添加 BOS/EOS
        tokens = self.tokenizer(
            str(sample['text']), 
            add_special_tokens=False, 
            max_length=self.max_length - 2,  # 为 BOS/EOS 保留 2 个位置
            truncation=True
        ).input_ids
        
        # 添加特殊标记
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # Padding：补齐到 max_length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Labels：用于计算预测损失
        # 在标准 LM 任务中：labels = input_ids（下一个 token 预测）
        labels = input_ids.clone()
        
        # 关键！将 padding 部分的标签设为 -100
        # 损失函数中 ignore_index=-100 会忽略这些位置
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return input_ids, labels


class SFTDataset(Dataset):
    """
    有监督微调（SFT）数据集
    
    格式：JSONL，每行包含多轮对话
    例子：{
        "conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么帮助吗？"}
        ]
    }
    
    关键特性：
    1. 使用 chat_template 格式化对话
    2. 只对 assistant 回复部分计算损失
    3. 支持工具调用（tool_call）和思考链（reasoning_content）
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 定义数据格式（用于验证）
        features = Features({
            'conversations': [{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),  # 思考链内容
                'tools': Value('string'),  # 工具定义
                'tool_calls': Value('string')  # 工具调用
            }]
        })
        
        # 加载数据
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        
        # 预计算 assistant 开始和结束标记
        # 用于在 generate_labels 中识别需要计算损失的部分
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        使用 tokenizer 的 chat_template 格式化对话
        
        输出格式（例）：
        <|im_start|>user
        你好
        <|im_end|>
        <|im_start|>assistant
        你好，有什么帮助吗？
        <|im_end|>
        """
        messages = []
        tools = None
        
        # 处理每条消息
        for message in conversations:
            message = dict(message)
            
            # 提取工具定义
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            
            # 解析 tool_calls（如果是字符串 JSON）
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            
            messages.append(message)
        
        # 应用 chat_template 进行格式化
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        生成损失掩码：只对 assistant 回复部分计算损失
        
        策略：
        1. 扫描输入序列，找到所有 <|im_start|>assistant\n 的位置
        2. 该位置到 <|im_end|> 之间的 token 的 labels = token 本身
        3. 其他部分的 labels = -100（忽略）
        
        这样做是因为：
        - 模型需要学习对用户输入做出正确的回复
        - 用户输入本身不需要学习预测
        """
        labels = [-100] * len(input_ids)  # 初始化为全忽略
        i = 0
        
        while i < len(input_ids):
            # 找到 <|im_start|>assistant\n 的位置
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)  # assistant 回复的起始位置
                end = start
                
                # 找到对应的 <|im_end|> 的位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # 标记这部分需要计算损失
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                
                # 继续扫描下一个对话轮次
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        
        return labels

    def __getitem__(self, index):
        """
        获取单个样本
        
        返回：(input_ids, labels)
        其中 labels 只在 assistant 部分非 -100
        """
        sample = self.samples[index]
        
        # 预处理对话（概率性添加 system）
        conversations = pre_processing_chat(sample['conversations'])
        
        # 格式化为 chat_template 格式
        prompt = self.create_chat_prompt(conversations)
        
        # 后处理（移除空思考标签）
        prompt = post_processing_chat(prompt)
        
        # Tokenize
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # Padding
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # 生成 labels（损失掩码）
        labels = self.generate_labels(input_ids)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """
    DPO（Direct Preference Optimization）数据集。

    数据格式要求每条样本包含两组对话：
    - chosen: 人类偏好的回答轨迹
    - rejected: 人类不偏好的回答轨迹

    训练时会分别计算 chosen/rejected 的 token logprob，
    然后通过 DPO 目标函数推动模型更偏向 chosen。
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        # 保存 tokenizer 和最大长度。
        self.tokenizer = tokenizer
        self.max_length = max_length
        # padding id，若 tokenizer 未显式设置则回退为 0。
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # assistant 段落的起止 token，用于构造 loss mask。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        # 从 json 数据文件加载训练样本。
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 取出单条偏好样本。
        sample = self.samples[index]
        chosen = sample['chosen']  # list[message]，偏好回答链路
        rejected = sample['rejected']  # list[message]，非偏好回答链路

        # 把 chosen/rejected 两条会话分别格式化成 chat prompt 文本。
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # 编码并固定长度 padding。
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # chosen 序列和对应 loss mask。
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        # rejected 序列和对应 loss mask。
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # 标准 next-token 训练格式：x 是输入，y 是右移一位标签。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """只在 assistant 回复区间打 1，其余位置为 0。"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到 assistant 开始标记。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 找到对应 assistant 结束标记。
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 标记 assistant 回复区间参与 loss。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """
    RLAIF 数据集。

    该数据集只返回 prompt，不直接返回训练标签。
    后续会在 rollout 阶段由策略模型在线生成回答，再根据 reward 进行优化。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 按概率开启 thinking，让策略同时覆盖思考/非思考两种输出模式。
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        # 下面两个字段当前类里未直接使用，保留是为了与其他数据集结构一致。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """从多轮对话构造 rollout 输入 prompt。"""
        # 先做通用预处理（如概率加 system prompt）。
        conversations = pre_processing_chat(conversations)
        # 随机决定本条样本是否开启 thinking。
        use_thinking = random.random() < self.thinking_ratio
        # 只保留到最后一条 user 前，add_generation_prompt=True 让模型继续生成 assistant。
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        # 返回 prompt；answer 留空，由训练时在线生成。
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    """
    Agent 强化学习数据集。

    样本中通常包含：
    - conversations: 历史消息（可能包含 tools）
    - gt: 任务期望结果（用于奖励计算）
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 这里直接手动逐行读取 JSONL，方便和上游数据格式完全对齐。
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        """解析对话并提取 tools 定义。"""
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # tools 可能挂在 system 消息里，且可能是 JSON 字符串。
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        # 最后一条通常是期望输出，不作为输入上下文。
        return messages[:-1], tools

    def __getitem__(self, index):
        # 返回 RL rollout 需要的 messages / tools / gt 三元组。
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass