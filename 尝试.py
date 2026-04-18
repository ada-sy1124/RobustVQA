from openai import OpenAI
import time

# 1. 建立与你私人显卡的连接
# 如果你的测试脚本和模型不在同一台机器，请把 127.0.0.1 换成模型机器的局域网 IP
API_BASE = "http://127.0.0.1:8000/v1"
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"

client = OpenAI(
    api_key="EMPTY",  # 本地服务不需要真实的 API Key
    base_url=API_BASE,
)

image_url = "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
user_prompt = "请仔细观察这张图片，告诉我图里有什么动物？它的表情看起来怎么样？并推测一下它可能在干什么？"

messages = [
    {
        "role": "system",
        "content": "你是一个幽默且观察力敏锐的视觉推理专家。"
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }
]

# 3. 发送请求并计时
# start_time = time.time()

try:
    # 就像调用真实的 OpenAI GPT-4o 一样调用你的本地模型
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.7,  # 稍微给一点创造力
        stream=True       # 开启流式输出（打字机效果），体验极佳！
    )

    
    # 接收流式数据，实现打字机效果
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
          
    # end_time = time.time()
    # print(f"\n⏱️ 推理完成！耗时: {end_time - start_time:.2f} 秒")

except Exception as e:
    print(f"\n❌ 调用失败，请检查网络或端口是否畅通。错误信息: {e}")