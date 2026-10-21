"""GSM8K 的数据格式、prompt 和判分。

数据行与 verl 的 examples/data_preprocess/gsm8k.py 产出的格式一致，第 ② 步可以直接复用：
  {"data_source": "openai/gsm8k",
   "prompt": [{"role": "user", "content": 题目 + " " + INSTRUCTION}],
   "ability": "math",
   "reward_model": {"style": "rule", "ground_truth": "72"},
   "extra_info": {"split": "train", "index": 0, "answer": 原始解答, "question": 原始题目}}

判分两种：
- strict：与 verl/utils/reward_score/gsm8k.py 的严格模式一致，训练奖励用它
- flexible：取回答里最后一个数，去掉千分位逗号后按数值比较，只在评测时用来拆分“格式错”和“算错”
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_SOURCE = "openai/gsm8k"
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'

# 与 verl 严格模式相同：只看最后 300 个字符，取最后一个“#### 数字”，去逗号后按字符串比较（2026-10-21 核对源码）
_STRICT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_CLIP_CHARS = 300
_NUMBER_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def extract_gold(answer_raw: str) -> str:
    """从 GSM8K 原始解答里取标准答案（与 verl 预处理相同）。"""
    match = _STRICT_RE.search(answer_raw)
    if match is None:
        raise ValueError(f"标准解答里没有 ####：{answer_raw[-80:]!r}")
    return match.group(1).replace(",", "")


def extract_strict(text: str) -> str | None:
    """取最后一个“#### 数字”里的数字；没写 #### 返回 None（按格式错误算 0 分）。"""
    matches = _STRICT_RE.findall(text[-_CLIP_CHARS:])
    return matches[-1].replace(",", "") if matches else None


def score_strict(response: str, gold: str) -> float:
    """训练奖励：答对 1.0，答错或没按格式写 0.0。"""
    return float(extract_strict(response) == gold)


def extract_flexible(text: str) -> str | None:
    """回答里最后一个数（去掉千分位逗号）；没有数字返回 None。"""
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def score_flexible(response: str, gold: str) -> bool:
    pred = extract_flexible(response)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def build_row(question: str, answer_raw: str, split: str, index: int) -> dict:
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": question + " " + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": extract_gold(answer_raw)},
        "extra_info": {"split": split, "index": index, "answer": answer_raw, "question": question},
    }


def encode_prompt(tokenizer, row: dict) -> list[int]:
    """chat template 加生成提示后分词，与 verl 的做法一致（不另加 special token）。"""
    text = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
