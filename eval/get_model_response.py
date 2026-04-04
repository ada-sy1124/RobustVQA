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

def extract_answer(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # 1
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    # 2
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.replace("<answer>", "").replace("</answer>", "")

    return text.strip()


def get_response(messages):
    client = openai.Client(base_url=f"http://{model_server}/v1", api_key="EMPTY")
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=model_path,
                messages=messages,
                temperature=0.7,
                max_tokens=4096
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
        max_workers = 512
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
