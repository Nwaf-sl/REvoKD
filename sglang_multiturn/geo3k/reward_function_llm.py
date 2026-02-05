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
import io
import logging
import os
import random
import re

import requests
from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

openai_api_key = "OTgzMWYwZGViZTIzZTNhYTU4ZjM2MDY2YWYxNWYyYmVjMGMxMDI2ZQ=="
openai_api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://1095312831785714.cn-shanghai.pai-eas.aliyuncs.com/api/predict/quickstart_deploy_20260131_zek6/v1")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
# models = client.models.list()
# print(models)
# model_name = ""
# if openai_api_base:
#     try:
#         response = requests.get(f"{openai_api_base}/models")
#         response.raise_for_status()
#         models = response.json()
#         if models.get("data"):
#             model_name = models["data"][0]["id"]
#         else:
#             logger.warning("No models found at the specified API base for reward scoring.")
#     except (requests.exceptions.RequestException, KeyError, IndexError) as e:
#         logger.warning(f"Failed to get model from {openai_api_base}: {e}. Reward scoring will be disabled.")
model_name = "Qwen3-30B-A3B-Instruct-2507"

logger.info(f"Using model: {model_name}")


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute reward score for model solutions with robust handling of various formats.

    Returns a weighted combination of:
    - Accuracy reward (0.8 weight): Whether the answer is semantically correct
    - Format reward (0.2 weight): Whether the output follows expected format
    - Tool reward (1.2 weight): Whether tools were used when answer is correct
    """
    save_dir = "all_solution_outputs"
    os.makedirs(save_dir, exist_ok=True)  # 创建目录
    save_file = os.path.join(save_dir, f"{data_source}_solutions.txt")

    # Initialize tracking variables
    is_format_error = False
    # print(f"[DEBUG] solution_str: {solution_str}")
    # # breakpoint()
    # print(f"[DEBUG] ground_truth: {ground_truth}")
   # 1. 提取 <answer>
    count_answer_1 = solution_str.count("<answer>")
    count_answer_2 = solution_str.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True
    
    # answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    # if answer_match:
    #     answer_text = answer_match.group(1).strip()
    # else:
    #     is_format_error = True
    #     answer_text = solution_str.strip()  # fallback
    
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if matches:
        answer_text = matches[-1].strip()  # 取最后一个
    else:
        is_format_error = True
        answer_text = solution_str.strip()  # fallback
    # 判空
    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()

    question_text = extra_info.get("question", "") if extra_info else ""

    # LLM 判定语义正确性
    if not client or not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0

    system_prompt = (
        "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
        "provided standard answer, given a specific question.\n"
        "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
        "standard answer.\n"
        'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
        "any explanation or other text."
    )

    user_prompt = (
        f"I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's "
        f"answer is correct.\n\n"
        f"---\n"
        f"**Example 1:**\n"
        f"[Question]: Is the countertop tan or blue?\n"
        f"[Standard Answer]: The countertop is tan.\n"
        f"[Model's Answer]: tan\n"
        f"[Your Judgement]: CORRECT\n"
        f"---\n"
        f"**Example 2:**\n"
        f"[Question]: Is the man phone both blue and closed?\n"
        f"[Standard Answer]: Yes, the man phone is both blue and closed.\n"
        f"[Model's Answer]: No.\n"
        f"[Your Judgement]: INCORRECT\n"
        f"---\n"
        f"**Task:**\n"
        f"[Question]: {question_text}\n"
        f"[Standard Answer]: {ground_truth}\n"
        f"[Model's Answer]: {answer_text}\n"
        f"[Your Judgement]:"
    )

    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=random.randint(0, 1000000),
            temperature=0.1,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        # print(f"[DEBUG] LLM response: {chat_response}")
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[WARNING] Chat completion request failed: {e}")
        return 0.0

    # 解析判定结果
    if re.search(r"\bCORRECT\b", response, re.IGNORECASE):
        acc_reward = 1.0
    elif re.search(r"\bINCORRECT\b", response, re.IGNORECASE):
        acc_reward = 0.0
    else:
        logger.warning(f"[WARNING] Invalid judgement: '{response}'")
        acc_reward = 0.0

    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    # 检查工具使用
    has_tool_usage = bool(
        re.search(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
        or re.search(r"<tool_response>.*?</tool_response>", solution_str, re.DOTALL)
    )

    # tool_reward = 1.0 if has_tool_usage and acc_reward > 0.5 else 0.0
    # if has_tool_usage:
    #     if acc_reward > 0.5:
    #         tool_reward = 1.0    # 用工具且准确度高
    #     else:
    #         tool_reward = 0.5    # 用工具但准确度低
    # else:
    #     tool_reward = 0.0        # 没用工具
    #======== new reward ========
    # 工具使用奖励
    if has_tool_usage:
        if acc_reward == 1.0:
            tool_reward = 1.0  # 正确 + 用工具 → 大加分
        else:
            tool_reward = 0.4  # 错误 + 用工具 → 小加分（鼓励探索）
    else:
        if acc_reward == 1.0:
            tool_reward = 0.2  # 正确但没用工具 → 小加分（但低于用工具）
        else:
            tool_reward = 0.0  # 错误且没用工具 → 零分



    
    # format_reward = -1.0 if is_format_error else 0.0
    format_reward = -0.5 if is_format_error else 0.5

    # final_score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
        # 来源加权系数（关键区别）
    # if data_source == "original":
    #     source_weight = 0.5   # 原始数据整体分数降低，避免模型过拟合旧知识
    # else:  # corrected
    #     source_weight = 1  # corrected 数据整体分数提高，强化新知识学习
    # 最终奖励
    final_score = acc_reward + format_reward + tool_reward
    # final_score = raw_score * source_weight
    # print(f"[DEBUG] final_score: {final_score}")
    # ======== 保存完整信息到txt ========
    with open(save_file, "a", encoding="utf-8") as f:
        f.write("==== new record ====\n")
        f.write(f"Question: {question_text}\n")
        f.write(f"Ground Truth: {ground_truth}\n")
        f.write(f"Solution: {solution_str}\n")
        f.write(f"Extracted Answer: {answer_text}\n")
        f.write(f"Final Score: {final_score}\n")
        f.write(f"Format Error: {is_format_error}\n")
        f.write(f"Has Tool Usage: {has_tool_usage}\n\n")

    return final_score



if __name__ == "__main__":
    # Test case 1: Original test case
    predict_str = "The answer is 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {
        "answer": "The woman is to the left of the man who is holding the camera.",
        "id": 0,
        "image": "/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg",
        "pred_ans": "The woman is to the right of the man who is holding the camera.",
        "question": "Is the woman to the left or to the right of the man who is holding the camera?",
    }
    print("=== Test Case 1: Original test ===")
    import time

    time_start = time.time()
    score = compute_score("common_reasoning", predict_str, ground_truth, extra_info)
    print(f"Score: {score}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 2: Problematic case mentioned by user
    problematic_solution = """<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [226, 399, 265, 464], "label": "white van"}}
</tool_call>user
<tool_response>
Zoomed in on the image to the region [226, 399, 265, 464] with label white van.
</tool_response>
assistant
The white van is visible in the lower section of the image, near the diagonal road."""

    problematic_ground_truth = "Yes, the white van is indeed situated in the bottom part of the picture."
    problematic_extra_info = {
        "question": "Is the white van in the bottom part of the picture?",
    }

    print("\n=== Test Case 2: Problematic case (no answer tags) ===")
    print(f"Solution: {problematic_solution}")
    print(f"Ground truth: {problematic_ground_truth}")

    time_start = time.time()
    score2 = compute_score("common_reasoning", problematic_solution, problematic_ground_truth, problematic_extra_info)
    print(f"Score: {score2}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 3: Well-formatted case with tools
    well_formatted_solution = """<think>
I need to use the image zoom tool to get a better look at the specific area.
</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [226, 399, 265, 464], "label": "white van"}}
</tool_call>
<tool_response>
Zoomed in on the image to the region [226, 399, 265, 464] with label white van.
</tool_response>
<answer>Yes, the white van is indeed situated in the bottom part of the picture.</answer>"""

    print("\n=== Test Case 3: Well-formatted case ===")
    time_start = time.time()
    score3 = compute_score(
        "common_reasoning", well_formatted_solution, problematic_ground_truth, problematic_extra_info
    )
    print(f"Score: {score3}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
