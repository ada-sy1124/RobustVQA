import random
import re
from tqdm import tqdm
import concurrent.futures
from verl.workers.reward_manager.utils.apis import request_verifier


verification_prompt_template="""You are required to act as a logic checker. Your task is to analyze the provided Reasoning and determine the correct option for the Question based **STRICTLY AND ONLY** on the information presented within that Reasoning.

Question and Options:
{question_and_options}

Reasoning to Analyze:
{cot}

---
Based on the provided Reasoning alone, what is the final answer? Your output MUST strictly follow the required format.

**STRICT OUTPUT FORMAT:**
<answer>
[The single letter corresponding to your final choice (A, B, C, or D). NOTHING ELSE.]
</answer>""".strip()


def shuffle_options_and_answer(original_options, correct_answer_letter):
    """
    随机打乱选项列表，确保打乱后的顺序与原始顺序差异尽量大，
    并返回打乱后的选项列表和对应的新标准答案
    """
    # 创建带索引的选项列表 [(0, A), (1, B), (2, C), (3, D)]
    indexed_options = list(enumerate(original_options))
    
    # 记录最佳打乱结果
    best_shuffle = None
    best_difference_score = -1

    N = len(original_options)
    
    # 尝试多次打乱，选择差异最大的那个
    for _ in range(100):
        # 随机打乱
        shuffled = indexed_options.copy()
        random.shuffle(shuffled)
        
        # 计算与原位置的差异程度
        # 每个元素不在原位置得1分，在原位置得0分
        difference_score = sum(1 for i, (orig_idx, _) in enumerate(shuffled) if orig_idx != i)
        
        # 如果这个打乱的差异更大，就更新最佳结果
        if difference_score > best_difference_score:
            best_difference_score = difference_score
            best_shuffle = shuffled.copy()
        
        # 如果已经达到最大差异（4个都不在原位置），就停止
        if difference_score == N:
            break
    
    # 解压最佳打乱结果，只保留选项内容
    shuffled_options = [option for _, option in best_shuffle]
    # 创建原始位置到字母的映射
    
    # position_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    position_to_letter = {i: chr(ord('A') + i) for i in range(N)}
    
    # 找到原始标准答案的位置
    original_position = ord(correct_answer_letter) - ord('A')  # A->0, B->1, C->2, D->3
    
    # 在打乱后的列表中查找原始标准答案内容的位置
    original_correct_content = original_options[original_position]  # 原始的标准答案内容
    new_position = shuffled_options.index(original_correct_content)  # 原始标准答案在打乱后列表中的位置
    
    # 获取新的答案字母
    new_answer_letter = position_to_letter[new_position]
    
    return shuffled_options, new_answer_letter


def create_vqa_prompt(question, choices):
    # 使用列表推导式动态生成 "A. 选项内容" 格式的列表
    options_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    # 拼接最终的 Prompt
    return f"Question: {question}\nOptions:\n{options_str}"


def make_prompt(question, cot, choices, ground_truth):
    # 打乱
    shuffled_options, new_answer = shuffle_options_and_answer(choices, ground_truth)
    # 重新生成prompt
    text = create_vqa_prompt(question, shuffled_options)
    # 这里替换
    N = len(choices)
    if N == 2:
        sub_text = "A or B"
    elif N == 3:
        sub_text = "A, B or C"
    elif N == 4:
        sub_text = "A, B, C, or D"
    elif N == 5:
        sub_text = "A, B, C, D, or E"
    else:
        raise ValueError("N error")

    _verification_prompt_template = verification_prompt_template.replace("A, B, C, or D", sub_text)
    prompt = _verification_prompt_template.format(
        question_and_options=text,
        cot=cot
    )
    return prompt, shuffled_options, new_answer


def get_format_reward(response):
    response = response.strip()
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    if not re.match(pattern, response, re.DOTALL):
        return -1.0
    
    if len(re.findall(r"<think>", response)) != 1 or len(re.findall(r"<answer>", response)) != 1:
        return -1.0

    return 0


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


def cot_length_reward(length, l_min, l_opt, l_max):
    if length < l_min:
        return -1
    elif length < l_opt:
        return -1 + (length - l_min) / (l_opt - l_min)
    elif length <= l_max:
        return 0
    else:
        return -1

def get_token_num(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    return len(tokens)


def get_length_penalty(items, tokenizer):
    length_rewards = []
    cot_lengths = []
    for item in items:
        cot = item["cot"]
        cot_len = get_token_num(cot, tokenizer)
        # 至少大于50，最佳为100
        reward = cot_length_reward(cot_len, 50, 100, 200)
        length_rewards.append(reward)
        cot_lengths.append(cot_len)
    
    answer_lengths = [get_token_num(item["response"], tokenizer) for item in items]
    
    return answer_lengths, cot_lengths, length_rewards

def process_reward(item):
    question = item["question"]
    cot = item["cot"]
    choices = item["choices"]
    ground_truth = item["ground_truth"]
    response = item["response"]

    # 
    N = len(choices)
    g_truth = [chr(ord('A') + i) for i in range(N)]

    ground_truth_text = choices[ord(ground_truth) - ord('A')]  # 标准答案的字符串
    if response in g_truth:
        response_text = choices[ord(response) - ord('A')]  # 第一次预测的字符串
    else:
        response_text = "T"
    
    prompt, shuffled_options, new_ground_truth = make_prompt(question, cot, choices, ground_truth)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    result = request_verifier(messages)
    shuffled_answer = extract_answer(result)
    if shuffled_answer in g_truth:
        shuffled_answer_text = shuffled_options[ord(shuffled_answer) - ord('A')]  # 打乱后的预测的字符串
    else:
        shuffled_answer_text = "F"
    # 计算reward
    # 1. acc reward
    if response == ground_truth:
        acc_reward = 1.0
    else:
        acc_reward = 0.0
    
    # consistency reward
    # 1. 两次都答对了
    if acc_reward == 1.0 and shuffled_answer == new_ground_truth:
        consistency_reward = 1.0
    # 第一次答对了，但打乱后没有答对
    elif acc_reward == 1.0 and shuffled_answer != new_ground_truth:
        consistency_reward = 0.5
    # 第一次没答对，纯文本反而答对了
    elif acc_reward == 0.0 and shuffled_answer == new_ground_truth:
        consistency_reward = 0.5  # 纯文本纠正了视觉错误
    # 两次都没对，但是回答的是一致的
    elif response_text == shuffled_answer_text:
        consistency_reward = 0.1
    else:
        consistency_reward = 0.0
    
    return acc_reward, consistency_reward


def get_rewards(items):
    # 并发
    max_workers = 512
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Start calculating rewards, number of tasks {len(items)}, number of concurrent workers {max_workers}")
        for item in items:
            futures.append(executor.submit(process_reward, item))
        
        futures = [future.result() for future in tqdm(futures)]
    
    acc_rewards, consistency_rewards = [], []
    for acc_reward, consistency_reward in futures:
        acc_rewards.append(acc_reward)
        consistency_rewards.append(consistency_reward)

    return acc_rewards, consistency_rewards




