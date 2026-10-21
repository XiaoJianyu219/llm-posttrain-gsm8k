"""带工具的贪心评测（第 ④ 步）：模型可写 Python、看执行结果、再答，多轮。

  python eval_tool.py --model <模型> --data <test.jsonl> --out-dir <目录> [--max-turns 3]

和 eval_gsm8k.py 同样报严格 / 宽松准确率，另外统计工具调用次数、失败率、轮数分布。
条件对照（no-tool / tool-no-train / tool-RL）里，带工具的那两格用这个脚本。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import extract_flexible, extract_strict, read_jsonl, score_flexible, score_strict
from tool_rollout import rollout_with_tools, set_tool_generation


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=400, help="每一轮的生成上限")
    ap.add_argument("--tool-timeout", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device=device, dtype=dtype).eval()
    gen = set_tool_generation(model, tokenizer, greedy=True, max_new_tokens=args.max_new_tokens)

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    questions = [r["extra_info"]["question"] for r in rows]
    start = time.perf_counter()
    trajs = rollout_with_tools(model, tokenizer, questions, device=device, gen_config=gen,
                               max_turns=args.max_turns, tool_timeout=args.tool_timeout)
    elapsed = time.perf_counter() - start

    samples = []
    for row, t in zip(rows, trajs):
        gold = row["reward_model"]["ground_truth"]
        samples.append({
            "split": row["extra_info"]["split"], "index": row["extra_info"]["index"], "gold": gold,
            "pred_strict": extract_strict(t.text), "pred_flexible": extract_flexible(t.text),
            "correct_strict": score_strict(t.text, gold) == 1.0, "correct_flexible": score_flexible(t.text, gold),
            "turns": t.turns, "tool_calls": t.tool_calls, "tool_errors": t.tool_errors,
            "finished": t.finished, "response_tokens": len(t.token_ids), "text": t.text,
        })
    n = len(samples)
    summary = {
        "model": args.model, "data": args.data, "n": n, "max_turns": args.max_turns,
        "acc_strict": sum(s["correct_strict"] for s in samples) / n,
        "acc_flexible": sum(s["correct_flexible"] for s in samples) / n,
        "format_rate": sum(s["pred_strict"] is not None for s in samples) / n,
        "tool_call_rate": sum(s["tool_calls"] > 0 for s in samples) / n,
        "tool_calls_mean": sum(s["tool_calls"] for s in samples) / n,
        "tool_error_rate": sum(s["tool_errors"] for s in samples) / max(1, sum(s["tool_calls"] for s in samples)),
        "turns_mean": sum(s["turns"] for s in samples) / n,
        "unfinished_rate": 1 - sum(s["finished"] for s in samples) / n,
        "elapsed_s": elapsed, "device": str(device),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n {n} | strict {summary['acc_strict']:.4f} | flex {summary['acc_flexible']:.4f} | "
          f"tool_call_rate {summary['tool_call_rate']:.3f} | tool_err {summary['tool_error_rate']:.3f} | "
          f"turns {summary['turns_mean']:.2f} | {elapsed:.0f}s")


if __name__ == "__main__":
    main()
