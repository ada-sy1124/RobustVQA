import json, random
from tqdm import tqdm
import os
import base64
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import shutil
import io


def get_prompt_template(N):
    letters = [chr(ord('A') + i) for i in range(N)]
    prompt_template = """Analyze the provided image and answer the following multiple-choice question.

    Your task is to first generate a step-by-step reasoning process, and then provide only the final chosen letter (A, B, C, or D).

    **STRICT OUTPUT FORMAT:**
    You must strictly adhere to the following structure:
    <think>
    [Your comprehensive, step-by-step reasoning process here.]
    </think>
    <answer>
    [The single letter corresponding to your final choice (A, B, C, or D). NOTHING ELSE.]
    </answer>

    {text}""".strip()
    # 替换
    # 2<=N<=5
    if N == 2:
        text = "A or B"
    elif N == 3:
        text = "A, B or C"
    elif N == 4:
        text = "A, B, C, or D"
    elif N == 5:
        text = "A, B, C, D, or E"
    else:
        raise ValueError("N error")

    return prompt_template.replace("A, B, C, or D", text)


def get_non_thinking_prompt_template(N):
    letters = [chr(ord('A') + i) for i in range(N)]
    prompt_template = """Analyze the provided image and answer the following multiple-choice question.

Your task is to provide only the final chosen letter (A, B, C, or D).

**STRICT OUTPUT FORMAT:**
You must strictly adhere to the following structure:
<answer>
[The single letter corresponding to your final choice (A, B, C, or D). NOTHING ELSE.]
</answer>

{text}""".strip()
    # 替换
    # 2<=N<=5
    if N == 2:
        text = "A or B"
    elif N == 3:
        text = "A, B or C"
    elif N == 4:
        text = "A, B, C, or D"
    elif N == 5:
        text = "A, B, C, D, or E"
    else:
        raise ValueError("N error")

    return prompt_template.replace("A, B, C, or D", text)


def create_vqa_prompt(question, choices):
    # 使用列表推导式动态生成 "A. 选项内容" 格式的列表
    options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    
    # 拼接最终的 Prompt
    return f"Question: {question}\nOptions:\n{options_str}"


def main(input_path, output_path):
    df = pd.read_parquet(input_path)
    datas = df.to_dict(orient='records')
    cnt = 0
    filter_datas = []
    all_num = []
    for data in datas:
        if data["image"] is not None:
            cnt += 1
            choices = data["choices"].tolist()
            if len(choices) >= 2:
                filter_datas.append(data)
                all_num.append(len(choices))
    
    print(min(all_num), max(all_num))
    # 图片数据
    all_data = []
    for data in filter_datas:
        choices = data["choices"].tolist()
        question = data["question"]
        ground_truth = chr(ord('A') + data["answer"]) # 0123形式，转为ABCD
        
        # 构造prompt
        text = create_vqa_prompt(question, choices)
        prompt_template = get_prompt_template(len(choices))
        text_prompt = prompt_template.format(text=text)
        
        content = "<image>" + text_prompt
        images = np.array([data["image"]], dtype=object)
        item = {
            "images": images,
            "prompt": [{"role": "user", "content": content}],
            "question": question,
            "ground_truth": ground_truth,
            "choices": choices
        }

        all_data.append(item)
    
    print(len(all_data))
    df = pd.DataFrame(all_data)
    df.to_parquet(output_path, index=False)


def generate_test_data(input_path, output_path):
    df = pd.read_parquet(input_path)
    datas = df.to_dict(orient='records')
    cnt = 0
    filter_datas = []
    all_num = []
    for data in datas:
        if data["image"] is not None:
            cnt += 1
            choices = data["choices"].tolist()
            if len(choices) >= 2:
                filter_datas.append(data)
                all_num.append(len(choices))
    
    print(min(all_num), max(all_num))
    # 图片数据
    all_data = []
    mime_type = "image/jpeg"
    for data in filter_datas:
        choices = data["choices"].tolist()
        question = data["question"]
        ground_truth = chr(ord('A') + data["answer"]) # 0123形式，转为ABCD
        
        # 构造prompt
        text = create_vqa_prompt(question, choices)
        prompt_template = get_prompt_template(len(choices))
        text_prompt = prompt_template.format(text=text)
        non_think_prompt_template = get_non_thinking_prompt_template(len(choices))
        non_thinking_text_prompt = non_think_prompt_template.format(text=text)
        # 编码图片
        base64_encoded_bytes = base64.b64encode(data["image"]["bytes"])
        base64_image = base64_encoded_bytes.decode('utf-8')
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
            {"type": "text", "text": text_prompt}
        ]
        non_content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
            {"type": "text", "text": non_thinking_text_prompt}
        ]
        item = {
            "prompt": [{"role": "user", "content": content}],
            "non_thinking_prompt": [{"role": "user", "content": non_content}],
            "question": question,
            "ground_truth": ground_truth,
            "choices": choices
        }

        all_data.append(item)
    
    print(len(all_data))
    with open(output_path, "w") as fout:
        for item in all_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main("../data/ScienceQA/train.parquet", "../data/scienceqa_train_data.parquet")
    main("../data/ScienceQA/test.parquet", "../data/scienceqa_test_data.parquet")

    generate_test_data("../data/ScienceQA/train.parquet", "../data/scienceqa_train_data.jsonl")
    generate_test_data("../data/ScienceQA/test.parquet", "../data/scienceqa_test_data.jsonl")
    