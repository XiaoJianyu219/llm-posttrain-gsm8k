"""P1 第 ③ 步 · 造 DPO 偏好对。

用训练前的模型（底座）对训练集每道题采 N 条回答，用和 GRPO 同一个严格判分打分；
同一道题里既有对的、也有错的，就随机挑一条对的当 chosen、一条错的当 rejected。
这样 DPO 和 GRPO 用的是同一个判分信号、同一个起点，区别只剩“离线”和“在线”。

  python make_dpo_pairs.py --model <底座> --data-dir <数据> --out <pairs.jsonl> [--backend vllm]

--backend hf 用训练里同一套 generate；--backend vllm 快得多，要在装了 vllm 的环境里跑。
两种后端采样参数相同：temperature 1.0，不做 top-k / top-p 截断，不加重复惩罚。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import encode_prompt, read_jsonl, score_flexible, score_strict, write_jsonl
from rollout import EOS_TOKENS, generate, set_generation


def sample_hf(args, tokenizer, prompts):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else
                          ("cpu" if args.device == "auto" else args.device))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device=device, dtype=dtype).eval()
    set_generation(model, tokenizer, greedy=False, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    torch.manual_seed(args.seed)
    expanded = [p for p in prompts for _ in range(args.samples_per_prompt)]
    responses, _ = generate(model, tokenizer, expanded, batch_size=args.gen_batch_size, device=device)
    return responses


def sample_vllm(args, tokenizer, prompts):
    from vllm import LLM, SamplingParams
    eos = [tokenizer.convert_tokens_to_ids(t) for t in EOS_TOKENS]
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=0.85)
    params = SamplingParams(n=args.samples_per_prompt, temperature=args.temperature, top_p=1.0, top_k=-1,
                            repetition_penalty=1.0, max_tokens=args.max_new_tokens, stop_token_ids=eos,
                            seed=args.seed)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    responses = []
    for out in outputs:
        for c in out.outputs:
            ids = list(c.token_ids)
            if c.finish_reason == "stop" and (not ids or ids[-1] not in eos):
                ids.append(eos[0])  # 和 hf 后端一致：正常结束的回答以 EOS 结尾
            responses.append(ids)
    return responses


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True, help="偏好对 jsonl；同目录另写一份 <名字>_stats.json")
    ap.add_argument("--samples-per-prompt", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gen-batch-size", type=int, default=512, help="只对 hf 后端有效")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--limit", type=int, default=0, help="只用训练集前 N 题（冒烟测试用），0 = 全部")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    rows = read_jsonl(Path(args.data_dir) / "train.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    prompts = [encode_prompt(tokenizer, r) for r in rows]
    golds = [r["reward_model"]["ground_truth"] for r in rows]
    n = args.samples_per_prompt

    start = time.perf_counter()
    responses = (sample_vllm if args.backend == "vllm" else sample_hf)(args, tokenizer, prompts)
    elapsed = time.perf_counter() - start
    texts = tokenizer.batch_decode(responses, skip_special_tokens=True)
    strict = [score_strict(t, golds[k // n]) for k, t in enumerate(texts)]
    flexible = [score_flexible(t, golds[k // n]) for k, t in enumerate(texts)]

    rng = random.Random(args.seed)
    pairs = []
    for i, row in enumerate(rows):
        good = [k for k in range(i * n, (i + 1) * n) if strict[k] == 1.0]
        bad = [k for k in range(i * n, (i + 1) * n) if strict[k] == 0.0]
        if good and bad:
            w, l = rng.choice(good), rng.choice(bad)
            pairs.append({"index": row["extra_info"]["index"], "gold": golds[i], "prompt_ids": prompts[i],
                          "chosen_ids": responses[w], "rejected_ids": responses[l],
                          "chosen": texts[w], "rejected": texts[l]})
    out = Path(args.out)
    write_jsonl(out, pairs)
    stats = {
        "backend": args.backend, "prompts": len(rows), "samples_per_prompt": n, "elapsed_s": elapsed,
        "strict_correct_rate": sum(strict) / len(strict), "flexible_correct_rate": sum(flexible) / len(flexible),
        "pairs": len(pairs),
        "chosen_tokens_mean": sum(len(p["chosen_ids"]) for p in pairs) / max(1, len(pairs)),
        "rejected_tokens_mean": sum(len(p["rejected_ids"]) for p in pairs) / max(1, len(pairs)),
    }
    (out.parent / f"{out.stem}_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(f"pairs {len(pairs)} / {len(rows)} prompts | strict {stats['strict_correct_rate']:.4f} "
          f"flex {stats['flexible_correct_rate']:.4f} | {elapsed:.0f}s ({args.backend})")


if __name__ == "__main__":
    main()
