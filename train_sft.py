"""P1 第 ③ 步 · SFT（监督微调）。

用 GSM8K 训练集的标准解答做模仿学习，只对回答部分算交叉熵，prompt 部分不算 loss。
标准解答里 <<48/2=24>> 这种计算器标注去掉，最后一行 "#### 72" 保留，结尾接 <|im_end|>，
这样学到的输出格式和训练、评测用的判分一致。prompt 和 GRPO 完全相同（chat template + #### 指令）。

  python train_sft.py --model <底座> --data-dir <prepare_data 的输出> --out-dir <运行目录>

输出：metrics.jsonl（每步一行）、val_stepXXXX.jsonl、run_summary.json、final/（bf16）。
测试集评测由 eval_gsm8k.py 做，和其他方法同一个脚本、同一个口径。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import encode_prompt, read_jsonl
from rollout import collate, forward_logprobs
from train_grpo import load_split, save_model, sync_weights, validate

CALC_RE = re.compile(r"<<[^>]*>>")


def build_examples(path: Path, tokenizer, max_prompt_len: int, max_response_len: int, limit: int = 0):
    """[(prompt_ids, target_ids)]：target 是去掉计算器标注的标准解答 + <|im_end|>。"""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    examples, dropped = [], 0
    for row in rows:
        prompt = encode_prompt(tokenizer, row)
        answer = CALC_RE.sub("", row["extra_info"]["answer"])
        target = tokenizer(answer, add_special_tokens=False)["input_ids"] + [im_end]
        if len(prompt) > max_prompt_len or len(target) > max_response_len:
            dropped += 1
            continue
        examples.append((prompt, target))
    return examples, dropped


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    """线性预热，之后余弦衰减到 0。"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64, help="每次参数更新用几条样本")
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-response-len", type=int, default=512, help="训练样本的长度上限，也是验证时的生成上限")
    ap.add_argument("--val-batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="只用训练集前 N 条（冒烟测试用），0 = 全部")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = torch.bfloat16 if device.type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    policy = AutoModelForCausalLM.from_pretrained(args.model).to(device=device, dtype=torch.float32)
    original_gen_config = copy.deepcopy(policy.generation_config)
    eval_copy = copy.deepcopy(policy).to(dtype=amp or torch.float32).eval().requires_grad_(False)
    if not args.no_grad_ckpt:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.config.use_cache = False
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  **({"fused": True} if device.type == "cuda" else {}))

    data_dir = Path(args.data_dir)
    examples, dropped = build_examples(data_dir / "train.jsonl", tokenizer, args.max_prompt_len,
                                       args.max_response_len, args.limit)
    val_items, _ = load_split(data_dir / "val.jsonl", tokenizer, args.max_prompt_len)
    val_args = SimpleNamespace(max_response_len=args.max_response_len, val_batch_size=args.val_batch_size)
    steps_per_epoch = len(examples) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    (out_dir / "run_config.json").write_text(json.dumps({
        "args": vars(args), "n_train": len(examples), "dropped_long": dropped, "total_steps": total_steps,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sft | train {len(examples)} (dropped {dropped}) | {total_steps} steps | lr {args.lr}")

    metrics_file = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    t_start = time.perf_counter()
    val0 = validate(eval_copy, tokenizer, val_items, val_args, device, 0, out_dir)
    metrics_file.write(json.dumps({"step": 0, **val0}) + "\n")
    print(f"step 0 | val strict {val0['val/acc_strict']:.3f} flex {val0['val/acc_flexible']:.3f}")
    val_history = [{"step": 0, **val0}]

    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        random.Random(args.seed + epoch).shuffle(order)
        policy.train()
        for b in range(steps_per_epoch):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            lr = lr_at(step, total_steps, warmup, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            n_tokens = sum(len(examples[i][1]) for i in idx)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for s in range(0, len(idx), args.micro_batch_size):
                chunk = idx[s:s + args.micro_batch_size]
                batch = collate([examples[i][0] for i in chunk], [examples[i][1] for i in chunk],
                                tokenizer.pad_token_id)
                logp, _ = forward_logprobs(policy, batch, device, amp)
                loss = -(logp * batch["response_mask"].to(device)).sum() / n_tokens  # 整批 token 平均的交叉熵
                loss.backward()
                loss_sum += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
            step += 1
            elapsed = time.perf_counter() - t0
            metrics_file.write(json.dumps({"step": step, "epoch": epoch, "train/loss": loss_sum, "train/lr": lr,
                                           "train/grad_norm": grad_norm.item(), "perf/time_per_step": elapsed,
                                           "perf/tokens_per_s": n_tokens / elapsed}) + "\n")
            if step % 10 == 0 or step == total_steps:
                metrics_file.flush()
                print(f"step {step}/{total_steps} | loss {loss_sum:.4f} | lr {lr:.2e} | {elapsed:.2f}s")
        sync_weights(policy, eval_copy)
        val = validate(eval_copy, tokenizer, val_items, val_args, device, step, out_dir)
        metrics_file.write(json.dumps({"step": step, "epoch": epoch, **val}) + "\n")
        metrics_file.flush()
        val_history.append({"step": step, **val})
        print(f"epoch {epoch} | val strict {val['val/acc_strict']:.3f} flex {val['val/acc_flexible']:.3f} "
              f"format {val['val/format_rate']:.3f}")

    save_model(eval_copy, tokenizer, original_gen_config, out_dir / "final")
    metrics_file.close()
    (out_dir / "run_summary.json").write_text(json.dumps({
        "steps": total_steps, "wall_clock_h": (time.perf_counter() - t_start) / 3600, "val_history": val_history,
        "final_model": str(out_dir / "final"), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done | final model: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
