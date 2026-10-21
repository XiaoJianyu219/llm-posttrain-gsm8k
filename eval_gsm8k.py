"""GSM8K 贪心评测：严格 / 宽松两种准确率，加逐题结果。

  python eval_gsm8k.py --model <模型目录或 HF 名> --data <data_dir>/test.jsonl --out-dir <输出目录>

严格：和训练奖励同一套判分（verl 严格模式）。宽松：取回答里最后一个数，按数值比较。
两者之差 = 算对了、但没按 “#### 答案” 格式写的题。
输出 samples.jsonl（逐题，compare_evals.py 用它做配对比较）和 summary.json。
第 ②、③ 步训练出来的模型（verl、SFT、DPO）也用这个脚本测，保证口径一致。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import encode_prompt, extract_flexible, extract_strict, read_jsonl, score_flexible, score_strict
from rollout import generate, set_generation


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl，prepare_data.py 的输出（如 test.jsonl）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="与训练的 --max-response-len 保持一致")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 题，0 = 全部")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device=device, dtype=dtype).eval()
    set_generation(model, tokenizer, greedy=True, max_new_tokens=args.max_new_tokens)

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    prompts = [encode_prompt(tokenizer, r) for r in rows]
    start = time.perf_counter()
    responses, finished = generate(model, tokenizer, prompts, batch_size=args.batch_size, device=device)
    elapsed = time.perf_counter() - start
    texts = tokenizer.batch_decode(responses, skip_special_tokens=True)

    samples = []
    for row, text, resp, fin in zip(rows, texts, responses, finished):
        gold = row["reward_model"]["ground_truth"]
        samples.append({
            "split": row["extra_info"]["split"], "index": row["extra_info"]["index"], "gold": gold,
            "pred_strict": extract_strict(text), "pred_flexible": extract_flexible(text),
            "correct_strict": score_strict(text, gold) == 1.0, "correct_flexible": score_flexible(text, gold),
            "finished": fin, "response_tokens": len(resp), "response": text,
        })

    n = len(samples)
    k_strict = sum(s["correct_strict"] for s in samples)
    k_flex = sum(s["correct_flexible"] for s in samples)
    summary = {
        "model": args.model, "data": args.data, "n": n,
        "acc_strict": k_strict / n, "acc_strict_ci95": wilson(k_strict, n),
        "acc_flexible": k_flex / n, "acc_flexible_ci95": wilson(k_flex, n),
        "format_rate": sum(s["pred_strict"] is not None for s in samples) / n,
        "clip_ratio": 1 - sum(s["finished"] for s in samples) / n,
        "response_tokens_mean": sum(s["response_tokens"] for s in samples) / n,
        "decoding": {"greedy": True, "repetition_penalty": 1.0, "max_new_tokens": args.max_new_tokens},
        "elapsed_s": elapsed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lo, hi = summary["acc_strict_ci95"]
    print(f"n {n} | acc_strict {summary['acc_strict']:.4f} (95% CI {lo:.4f}-{hi:.4f}) | "
          f"acc_flexible {summary['acc_flexible']:.4f} | format_rate {summary['format_rate']:.4f} | "
          f"clip_ratio {summary['clip_ratio']:.4f} | len {summary['response_tokens_mean']:.0f} | {elapsed:.0f}s")


if __name__ == "__main__":
    main()
