# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.utils.rewards import get_rewards, get_length_penalty

import json
import re
import numpy as np


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


@register("custom")
class CustomRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        valid_response_lengths = []

        format_rewards = []
        score_data = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            format_reward = get_format_reward(response_str)
            format_rewards.append(format_reward)

            cot = response_str.split("</think>")[0].split("<think>")[-1].strip()

            # think
            response = extract_answer(response_str)
            item = {
                "question": data_item.non_tensor_batch.get("question"),
                "cot": cot,
                "response": response,
                "ground_truth": data_item.non_tensor_batch.get("ground_truth"),
                "choices": data_item.non_tensor_batch.get("choices")
            }
            score_data.append(item)

            # print("cot:", cot)
            # print("response:", response)

            valid_response_lengths.append(valid_response_length)
        

        print("calculating rewards...")
        acc_rewards, consistency_rewards = get_rewards(score_data)
        print("Rewards calculated.")

        # 思维链长度惩罚，鼓励思考
        print("calculating length scores...")
        answer_lengths, cot_lengths, length_rewards = get_length_penalty(score_data, self.tokenizer)
        print("length scores calculated.")

        for i in range(len(data)):
            acc_reward = acc_rewards[i]
            consistency_reward = consistency_rewards[i]
            format_reward = format_rewards[i]
            length_reward = length_rewards[i]

            # 如果回答正确就不鼓励冗余思考
            if acc_reward == 1.0 and consistency_reward == 1.0:
                length_reward = 0.0
                length_rewards[i] = 0.0

            total_reward = 0.7 * acc_reward + 0.3 * consistency_reward + format_reward + length_reward
            reward_tensor[i, valid_response_lengths[i] - 1] = total_reward

        return reward_tensor, acc_rewards, consistency_rewards, format_rewards, length_rewards, cot_lengths, answer_lengths
