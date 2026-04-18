import os
import json
import re
import requests
import json
import time
import openai
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import argparse


model_server = os.environ.get('MODEL_SERVER')
model_path = os.environ.get('MODEL_PATH')


# =================================================================================================
# 清洗模型的输出，从无论多恶心的输出里提取出对应的答案ABCD。。。
# =================================================================================================
def extract_answer(text: str) -> str:
    if not text or not isinstance(text, str):  #如果没有输出结果就返回""
        return ""
    # 1. 
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    # 2
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("<answer>", "").replace("</answer>", "")

    return text.strip()

# =================================================================================================
# API请求测试，这个咱门儿清，街景里经常用到，因为调用Gemini API经常403/502报错
# 导致服务器短时间的“堵塞”、超时报错（Timeout）等等，我们会最多重试5次，等待时间是1秒，如果5次都失败就回复空""
# =================================================================================================
def get_response(messages):
    client = openai.Client(base_url=f"http://{model_server}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=model_path,
                messages=messages,
                temperature=0.7, ############################
                max_tokens=4096 ############################
            )
            return response.choices[0].message.content.strip()

        except:
            time.sleep(1) 
            continue
    
    return ""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RobustVQA测评")

    # 添加参数
    parser.add_argument("--model_id", type=str, default="Qwen2.5-VL-7B-Instruct", help="模型名")  
    parser.add_argument("--dataset", type=str, default="", help="数据集名称")

    args = parser.parse_args()

    with open(f"./dataset/{args.dataset}", "r") as fin:
        datas = [json.loads(line) for line in fin]
    
    with open(f"./output/{args.dataset}.{args.model_id}.output.jsonl", "w") as fout:
        max_workers = 512 # 512线程并发
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for data in datas:
                # 处理images
                if "RL" in args.model_id:
                    prompt = data["prompt"]
                    futures.append(executor.submit(get_response, prompt))
                else:
                    prompt = data["non_thinking_prompt"]
                    futures.append(executor.submit(get_response, prompt))

            pairs = [(data, future) for data, future in zip(datas, futures)]

            for data, future in tqdm(pairs):
                response = future.result()
                data["response"] = response
                data["answer"] = extract_answer(response)
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")


# 经过这段代码的运行，样本结果变成以下格式：

# {
#   "prompt": [
#     {
#       "role": "user",
#       "content": [
#         {
#           "type": "image_url",
#           "image_url": { "url": "" }
#         },
#         {
#           "type": "text",
#           "text": "Analyze the provided image and answer the following multiple-choice question...\nYour task is to provide only the final chosen letter (A, B or C)..."
#         }
#       ]
#     }
#   ],
#   "question": "Which of the following could Gordon's test show?",
#   "ground_truth": "B",
#   "choices": [
#     "if the spacecraft was damaged when using a parachute with a 1 m vent going 200 km per hour",
#     "how steady a parachute with a 1 m vent was at 200 km per hour",
#     "whether a parachute with a 1 m vent would swing too much at 400 km per hour"
#   ],
#   "response": "<answer>\nB\n</answer>", ##########################################
#   "answer": "B" ##############################################
# }