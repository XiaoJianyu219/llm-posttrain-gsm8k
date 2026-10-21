"""P1 第 ③ 步 · DPO。

L = −log σ(β·[(log π_θ(y_w) − log π_ref(y_w)) − (log π_θ(y_l) − log π_ref(y_l))])
log π 是整条回答所有 token 的对数概率之和；参考模型就是训练前的底座，训练开始前一次算好。
另外记录 chosen / rejected 相对参考模型各自涨跌多少：常见现象是两者一起下降，只是 rejected 降得更多。

  python train_dpo.py --model <底座> --pairs <pairs.jsonl> --data-dir <数据> --out-dir <运行目录>

输出：metrics.jsonl、val_stepXXXX.jsonl、run_summary.json、final/（bf16）。
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import read_jsonl
from rollout import collate, forward_logprobs
from train_grpo import load_split, save_model, sync_weights, validate
from train_sft import lr_at


def sequence_logps(model, pairs: list, tokenizer, device, amp):
    """一批偏好对 → (chosen 的 log π 之和, rejected 的 log π 之和)，chosen 和 rejected 拼在一个 batch 里算。"""
    prompts = [p["prompt_ids"] for p in pairs] * 2
    responses = [p["chosen_ids"] for p in pairs] + [p["rejected_ids"] for p in pairs]
    batch = collate(prompts, responses, tokenizer.pad_token_id)
    logp, _ = forward_logprobs(model, batch, device, amp)
    sums = (logp * batch["response_mask"].to(device)).sum(dim=-1)
    return sums[:len(pairs)], sums[len(pairs):]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True, help="用里面的 val.jsonl 做验证")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32, help="每次参数更新用几对")
    ap.add_argument("--micro-batch-size", type=int, default=4, help="每次前向几对（即 2 倍条数）")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-response-len", type=int, default=512, help="验证时的生成上限")
    ap.add_argument("--val-batch-size", type=int, default=256)
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
    pairs = read_jsonl(args.pairs)
    if not pairs:
        raise SystemExit(f"{args.pairs} 里没有偏好对")
    val_items, _ = load_split(Path(args.data_dir) / "val.jsonl", tokenizer, args.max_prompt_len)
    val_args = SimpleNamespace(max_response_len=args.max_response_len, val_batch_size=args.val_batch_size)

    # 参考模型的 log π：训练开始前用底座一次算好，之后不再变
    policy.eval()
    ref_w, ref_l = [], []
    with torch.no_grad():
        for s in range(0, len(pairs), args.micro_batch_size):
            w, l = sequence_logps(policy, pairs[s:s + args.micro_batch_size], tokenizer, device, amp)
            ref_w.append(w.float().cpu())
            ref_l.append(l.float().cpu())
    ref_w, ref_l = torch.cat(ref_w), torch.cat(ref_l)

    if not args.no_grad_ckpt:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.config.use_cache = False
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  **({"fused": True} if device.type == "cuda" else {}))
    steps_per_epoch = max(1, len(pairs) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    (out_dir / "run_config.json").write_text(json.dumps({
        "args": vars(args), "n_pairs": len(pairs), "total_steps": total_steps,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dpo | pairs {len(pairs)} | {total_steps} steps | beta {args.beta} lr {args.lr}")

    metrics_file = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    t_start = time.perf_counter()
    val0 = validate(eval_copy, tokenizer, val_items, val_args, device, 0, out_dir)
    metrics_file.write(json.dumps({"step": 0, **val0}) + "\n")
    print(f"step 0 | val strict {val0['val/acc_strict']:.3f} flex {val0['val/acc_flexible']:.3f}")
    val_history = [{"step": 0, **val0}]

    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        random.Random(args.seed + epoch).shuffle(order)
        policy.train()
        for b in range(steps_per_epoch):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            lr = lr_at(step, total_steps, warmup, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            st = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "chosen_delta": 0.0, "rejected_delta": 0.0}
            for s in range(0, len(idx), args.micro_batch_size):
                chunk = idx[s:s + args.micro_batch_size]
                pw, pl = sequence_logps(policy, [pairs[i] for i in chunk], tokenizer, device, amp)
                dw = pw - ref_w[chunk].to(device)  # chosen 相对参考模型的 log π 变化
                dl = pl - ref_l[chunk].to(device)
                margin = args.beta * (dw - dl)
                loss = -F.logsigmoid(margin).sum() / len(idx)
                loss.backward()
                st["loss"] += loss.item()
                st["acc"] += (margin > 0).float().sum().item() / len(idx)
                st["margin"] += margin.sum().item() / len(idx)
                st["chosen_delta"] += dw.sum().item() / len(idx)
                st["rejected_delta"] += dl.sum().item() / len(idx)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
            step += 1
            metrics_file.write(json.dumps({"step": step, "epoch": epoch, "train/lr": lr,
                                           "train/grad_norm": grad_norm.item(),
                                           **{f"train/{k}": v for k, v in st.items()}}) + "\n")
            metrics_file.flush()
            print(f"step {step}/{total_steps} | loss {st['loss']:.4f} acc {st['acc']:.2f} "
                  f"chosen {st['chosen_delta']:+.2f} rejected {st['rejected_delta']:+.2f}")
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
        "steps": total_steps, "pairs": len(pairs), "wall_clock_h": (time.perf_counter() - t_start) / 3600,
        "val_history": val_history, "final_model": str(out_dir / "final"),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done | final model: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
