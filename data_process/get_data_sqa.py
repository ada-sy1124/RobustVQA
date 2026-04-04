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


# image : 上下文图片（为问题提供视觉背景信息）
# question : 任务问题
# choices : 选项，其中包含正确答案
# answer : 正确答案在 choices 列表中的索引
# hint : 帮助回答任务问题的提示
# task : 任务描述
# grade : 本题难度 对应的 K-12（幼儿园至高中）
# subject : 学科大类
# topic : 课题分类（具体为：自然科学、社会科学 或 语言科学）
# category : 课题（topic）下的细分子类别
# skill : 解决该问题所需技能的描述
# lecture : 生成该问题所依据的背景知识
# solution : 如何解答问题的instruction


# ==============================================================================================
# get_prompt_template函数，输入是choices的数量，输出是对应的prompt
# 因为不同选项数量的abcd数不同，于是这个函数为不同的choices数输出略有不同的prompt
# ==============================================================================================
def get_prompt_template(N):
    # letters = [chr(ord('A') + i) for i in range(N)]
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


# ==============================================================================================
# 以上要求思维链版本的对应版本，无思维链的版本，直接让模型判断结果
# ==============================================================================================
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


# ==============================================================================================
# create_vqa_prompt函数，输入ScienceQA的 question 和 choices 列，将其拼凑为带ABCD格式的 QA prompt
# 输入 question = "地球的卫星是哪个？"，choices = ["太阳", "月亮", "火星"]
# 输出 ：
# Question: 地球的卫星是哪个？
# Options:
# A. 太阳
# B. 月亮
# C. 火星
# ==============================================================================================
def create_vqa_prompt(question, choices):
    # 使用列表推导式动态生成 "A. 选项内容" 格式的列表
    options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    
    # 拼接最终的 Prompt
    return f"Question: {question}\nOptions:\n{options_str}"


# ==============================================================================================
# 数据集格式化函数
# 将ScienceQA 数据集转换成可用于训练的标准数据格式
# 包括：剔除无image的样本、提取ground truth并将 anwser 转为ABCD的格式、将问题组装成标准训练格式等
# ==============================================================================================
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
    