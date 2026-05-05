"""最小化的 OpenAI 兼容聊天客户端示例。

这个脚本用于直接连接本地兼容 OpenAI 接口的服务，做交互式问答。
"""

from openai import OpenAI

# 初始化一个 OpenAI 客户端，但 base_url 指向本地服务，而不是官方接口。
client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:11434/v1"
)

# 是否启用流式输出。
stream = True

# 原始对话历史模板，当前脚本里先从空列表开始。
conversation_history_origin = []

# 实际对话历史，复制一份，避免直接修改原始模板对象。
conversation_history = conversation_history_origin.copy()

# 历史消息数量，必须是偶数，表示保留多少轮 Q+A。
# 设为 0 时表示不携带历史，只发当前轮内容。
history_messages_num = 0

# 持续交互，直到用户手动中断。
while True:
    # 读取一条用户输入。
    query = input('[Q]: ')

    # 把用户问题追加到历史记录中。
    conversation_history.append({"role": "user", "content": query})

    # 请求本地兼容 OpenAI 的聊天接口。
    response = client.chat.completions.create(
        model="minimind-local:latest",
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.8,
        # 通过额外参数打开思考模式，并指定 reasoning 强度。
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"}
    )

    # 非流式输出时，直接读取完整回答。
    if not stream:
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        # 流式输出时，边收到边打印，提升交互体验。
        print('[A]: ', end='', flush=True)
        assistant_res = ''
        for chunk in response:
            delta = chunk.choices[0].delta
            # reasoning_content 是思考内容，content 是最终回答正文。
            r = getattr(delta, 'reasoning_content', None) or ""
            c = delta.content or ""
            if r:
                # 用灰色打印思考内容，和正文做视觉区分。
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:
                print(c, end="", flush=True)
            assistant_res += c

    # 把模型回答写回历史，供下一轮上下文使用。
    conversation_history.append({"role": "assistant", "content": assistant_res})

    # 每轮结束后空两行，保持终端可读性。
    print('\n\n')