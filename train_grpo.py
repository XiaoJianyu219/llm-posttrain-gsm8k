"""P1 第 ① 步：自写 GRPO（GSM8K + Qwen2.5-0.5B-Instruct，单卡）。

每一步按 verl 的顺序分段计时：
  gen（采样）→ reward（判分）→ old_log_prob（训练端重算 logprob）→ ref（参考模型）→ adv（组内优势）
  → update_actor（按 mini-batch 更新）→ update_weights（新权重拷进采样用的 bf16 副本）
段名和指标名尽量照 verl 的日志取（timing_s/*、critic/score/mean、actor/pg_loss 等），第 ② 步可以逐项对比。

  python train_grpo.py --model <模型目录或 HF 名> --data-dir <prepare_data 的输出> --out-dir <本次运行目录>

输出：metrics.jsonl（每步一行）、samples.jsonl（每步几条采样）、val_stepXXXX.jsonl（验证集逐题结果）、
run_config.json、run_summary.json、final/（训练后的模型，bf16）、checkpoints/。
测试集不在这里用：训练前、训练后各测一次，由 eval_gsm8k.py 做。
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_core import agg_loss, grpo_advantages, kl_k3, policy_loss
from gsm8k import encode_prompt, extract_strict, read_jsonl, score_flexible, score_strict
from rollout import collate, forward_logprobs, generate, set_generation


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data-dir", required=True, help="prepare_data.py 的输出目录（train.jsonl、val.jsonl）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=150)
    # 括号里是 verl 中对应的配置项，第 ② 步照这些值设
    ap.add_argument("--train-batch-size", type=int, default=64, help="每步几道题（data.train_batch_size）")
    ap.add_argument("--group-size", type=int, default=8, help="每道题采几条回答（rollout.n）")
    ap.add_argument("--mini-batch-size", type=int, default=16, help="每次参数更新用几道题（ppo_mini_batch_size）")
    ap.add_argument("--micro-batch-size", type=int, default=4, help="每次前向几条回答（ppo_micro_batch_size_per_gpu）")
    ap.add_argument("--gen-batch-size", type=int, default=256, help="每次 generate 几条回答")
    ap.add_argument("--max-prompt-len", type=int, default=512, help="data.max_prompt_length")
    ap.add_argument("--max-response-len", type=int, default=512, help="data.max_response_length")
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout.temperature")
    ap.add_argument("--lr", type=float, default=1e-6, help="actor.optim.lr")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="actor.optim.weight_decay")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="actor.grad_clip")
    ap.add_argument("--clip-ratio", type=float, default=0.2, help="actor.clip_ratio_low")
    ap.add_argument("--clip-ratio-high", type=float, default=None, help="actor.clip_ratio_high，不设则同 --clip-ratio")
    ap.add_argument("--clip-ratio-c", type=float, default=3.0, help="actor.clip_ratio_c（dual-clip）")
    ap.add_argument("--kl-coef", type=float, default=0.001,
                    help="actor.kl_loss_coef（use_kl_loss=True、low_var_kl）；0 = 不用 KL，也不加载参考模型")
    ap.add_argument("--loss-agg-mode", default="token-mean", choices=["token-mean", "seq-mean-token-mean"],
                    help="actor.loss_agg_mode")
    ap.add_argument("--no-std-norm", action="store_true",
                    help="优势只减组内均值（Dr. GRPO；algorithm.norm_adv_by_std_in_grpo=False）")
    ap.add_argument("--val-every", type=int, default=25, help="每隔几步在验证集上贪心评测（trainer.test_freq）")
    ap.add_argument("--val-limit", type=int, default=0, help="只用验证集前 N 题，0 = 全部")
    ap.add_argument("--val-batch-size", type=int, default=128)
    ap.add_argument("--save-every", type=int, default=50, help="每隔几步存一次权重（bf16，约 1 GB），0 = 只存最终")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="关掉梯度检查点（更快，但更占显存）")
    ap.add_argument("--debug-reward", choices=["gsm8k", "alternate"], default="gsm8k",
                    help="alternate 只给冒烟测试用：每组内交替给 1/0，保证每组都有梯度")
    return ap.parse_args()


@contextmanager
def timed(name: str, timing: dict, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize()
    timing[name] = timing.get(name, 0.0) + time.perf_counter() - start


def load_split(path: Path, tokenizer, max_prompt_len: int):
    """返回 [(prompt_ids, gold, question)]，以及因 prompt 太长被丢掉的条数。"""
    items, dropped = [], 0
    for row in read_jsonl(path):
        ids = encode_prompt(tokenizer, row)
        if len(ids) > max_prompt_len:
            dropped += 1
            continue
        items.append((ids, row["reward_model"]["ground_truth"], row["extra_info"]["question"]))
    return items, dropped


def prompt_batches(n_items: int, batch_size: int, seed: int):
    """每个 epoch 重新打乱，凑不满一批的丢掉。产出 (epoch, 题目下标列表)。"""
    epoch = 0
    while True:
        order = list(range(n_items))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, n_items - batch_size + 1, batch_size):
            yield epoch, order[start:start + batch_size]
        epoch += 1


@torch.no_grad()
def sync_weights(src, dst) -> None:
    """update_weights：把训练端的 fp32 主权重拷进采样副本（GPU 上是 bf16）。"""
    for p_dst, p_src in zip(dst.parameters(), src.parameters()):
        p_dst.copy_(p_src)


def validate(model, tokenizer, items, args, device, step: int, out_dir: Path) -> dict:
    """验证集贪心评测，逐题结果写 val_stepXXXX.jsonl。"""
    set_generation(model, tokenizer, greedy=True, max_new_tokens=args.max_response_len)
    responses, finished = generate(model, tokenizer, [it[0] for it in items],
                                   batch_size=args.val_batch_size, device=device)
    texts = tokenizer.batch_decode(responses, skip_special_tokens=True)
    rows = []
    for (_, gold, question), text, resp, fin in zip(items, texts, responses, finished):
        rows.append({
            "question": question, "gold": gold, "response": text,
            "correct_strict": score_strict(text, gold) == 1.0,
            "correct_flexible": score_flexible(text, gold),
            "has_format": extract_strict(text) is not None,
            "finished": fin, "response_tokens": len(resp),
        })
    with open(out_dir / f"val_step{step:04d}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(rows)
    return {
        "val/acc_strict": sum(r["correct_strict"] for r in rows) / n,
        "val/acc_flexible": sum(r["correct_flexible"] for r in rows) / n,
        "val/format_rate": sum(r["has_format"] for r in rows) / n,
        "val/response_length/mean": sum(r["response_tokens"] for r in rows) / n,
        "val/response_length/clip_ratio": 1 - sum(r["finished"] for r in rows) / n,
    }


def save_model(model, tokenizer, gen_config, path: Path) -> None:
    """保存采样副本（已与训练端同步；GPU 上是 bf16，约 1 GB），generation_config 恢复成模型原来的。"""
    current = model.generation_config
    model.generation_config = gen_config
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    model.generation_config = current


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    args = parse_args()
    G = args.group_size
    if args.train_batch_size % args.mini_batch_size:
        raise SystemExit("--train-batch-size 必须是 --mini-batch-size 的整数倍")
    if (args.mini_batch_size * G) % args.micro_batch_size:
        raise SystemExit("--mini-batch-size × --group-size 必须是 --micro-batch-size 的整数倍")
    clip_high = args.clip_ratio if args.clip_ratio_high is None else args.clip_ratio_high
    n_mini = args.train_batch_size // args.mini_batch_size
    seqs_per_mini = args.mini_batch_size * G
    micro_per_mini = seqs_per_mini // args.micro_batch_size

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = torch.bfloat16 if device.type == "cuda" else None  # CPU（冒烟测试）全程 fp32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    # 训练端：fp32 主权重，前向在 bf16 autocast 下算（相当于 verl 里 FSDP 的混合精度）
    policy = AutoModelForCausalLM.from_pretrained(args.model).to(device=device, dtype=torch.float32)
    original_gen_config = copy.deepcopy(policy.generation_config)
    # 参考模型：初始权重的冻结副本，与训练端同精度、同算法，所以第 1 步的 KL 应该是 0
    ref = copy.deepcopy(policy).eval().requires_grad_(False) if args.kl_coef > 0 else None
    # 采样副本：相当于 verl 里的推理引擎，每步更新完同步一次权重
    actor_rollout = copy.deepcopy(policy).to(dtype=amp or torch.float32).eval().requires_grad_(False)
    if not args.no_grad_ckpt:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.config.use_cache = False
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                                  **({"fused": True} if device.type == "cuda" else {}))

    data_dir = Path(args.data_dir)
    train_items, dropped_train = load_split(data_dir / "train.jsonl", tokenizer, args.max_prompt_len)
    val_items, dropped_val = load_split(data_dir / "val.jsonl", tokenizer, args.max_prompt_len)
    if args.val_limit:
        val_items = val_items[:args.val_limit]
    split_info = data_dir / "split_info.json"
    run_config = {
        "args": vars(args),
        "clip_ratio_high": clip_high,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "cuda": torch.version.cuda},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_train": len(train_items), "n_val": len(val_items),
        "dropped_long_prompts": {"train": dropped_train, "val": dropped_val},
        "split_info": json.loads(split_info.read_text(encoding="utf-8")) if split_info.exists() else None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device {run_config['device']} | train {len(train_items)} val {len(val_items)} | "
          f"{args.steps} steps x {args.train_batch_size} prompts x {G} samples | "
          f"{n_mini} updates/step | kl_coef {args.kl_coef} | loss_agg {args.loss_agg_mode}")

    metrics_file = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    samples_file = open(out_dir / "samples.jsonl", "a", encoding="utf-8")
    t_start = time.perf_counter()

    # 训练前先在验证集上测一次（verl 的 val_before_train）
    timing0: dict = {}
    with timed("testing", timing0, device):
        val0 = validate(actor_rollout, tokenizer, val_items, args, device, 0, out_dir)
    metrics_file.write(json.dumps({"step": 0, **val0, "timing_s/testing": timing0["testing"]}) + "\n")
    metrics_file.flush()
    print(f"step 0 | val strict {val0['val/acc_strict']:.3f} flex {val0['val/acc_flexible']:.3f} "
          f"format {val0['val/format_rate']:.3f} | {timing0['testing']:.0f}s")
    val_history = [{"step": 0, **val0}]

    batches = prompt_batches(len(train_items), args.train_batch_size, args.seed)
    timing_sums: dict = {}
    recent_steps: list = []
    for step in range(1, args.steps + 1):
        timing: dict = {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch, picked = next(batches)
        prompts = [train_items[i][0] for i in picked for _ in range(G)]
        golds = [train_items[i][1] for i in picked for _ in range(G)]
        n = len(prompts)

        with timed("step", timing, device):
            with timed("gen", timing, device):
                set_generation(actor_rollout, tokenizer, greedy=False, max_new_tokens=args.max_response_len,
                               temperature=args.temperature)
                responses, finished = generate(actor_rollout, tokenizer, prompts,
                                               batch_size=args.gen_batch_size, device=device)
            if device.type == "cuda":
                torch.cuda.empty_cache()

            with timed("reward", timing, device):
                texts = tokenizer.batch_decode(responses, skip_special_tokens=True)
                if args.debug_reward == "alternate":
                    rewards = (torch.arange(n) % 2).float()
                else:
                    rewards = torch.tensor([score_strict(t, g) for t, g in zip(texts, golds)])

            # old_log_prob、ref、更新三处用同一套 micro-batch 划分，保证数值可比
            micro = [list(range(s, s + args.micro_batch_size)) for s in range(0, n, args.micro_batch_size)]
            mbs = [collate([prompts[i] for i in idx], [responses[i] for i in idx], tokenizer.pad_token_id)
                   for idx in micro]

            with timed("old_log_prob", timing, device):
                policy.eval()
                old_logp, ent_sum, tok_sum = [], 0.0, 0.0
                with torch.no_grad():
                    for b in mbs:
                        logp, ent = forward_logprobs(policy, b, device, amp, with_entropy=True)
                        mask = b["response_mask"].to(device)
                        old_logp.append(logp)
                        ent_sum += (ent * mask).sum().item()
                        tok_sum += mask.sum().item()

            # 训推不一致的小测量：同一批回答，采样副本（bf16）和训练端（fp32 主权重 + autocast）算出的 logprob 差多少
            with timed("diag", timing, device), torch.no_grad():
                rollout_logp, _ = forward_logprobs(actor_rollout, mbs[0], device, None)
                mask0 = mbs[0]["response_mask"].to(device)
                gap = (rollout_logp - old_logp[0]).abs() * mask0
                gap_mean = (gap.sum() / mask0.sum()).item()
                gap_max = gap.max().item()

            ref_logp = None
            if ref is not None:
                with timed("ref", timing, device), torch.no_grad():
                    ref_logp = [forward_logprobs(ref, b, device, amp)[0] for b in mbs]

            with timed("adv", timing, device):
                adv = grpo_advantages(rewards, G, norm_by_std=not args.no_std_norm).to(device)

            with timed("update_actor", timing, device):
                policy.train()
                st = {"pg_loss": 0.0, "kl_loss": 0.0, "grad_norm": 0.0, "clipped": 0.0, "dual_clipped": 0.0,
                      "neg_log_ratio": 0.0, "tokens": 0.0, "skipped": 0}
                ratio_maxdev = None
                for m in range(n_mini):
                    ids = range(m * micro_per_mini, (m + 1) * micro_per_mini)
                    n_tokens = sum(mbs[j]["response_mask"].sum().item() for j in ids)
                    optimizer.zero_grad(set_to_none=True)
                    for j in ids:
                        b = mbs[j]
                        mask = b["response_mask"].to(device)
                        logp, _ = forward_logprobs(policy, b, device, amp)
                        tok_adv = adv[micro[j]].unsqueeze(-1).expand_as(logp)
                        pg_mat, counts = policy_loss(logp, old_logp[j], tok_adv, mask, clip_low=args.clip_ratio,
                                                     clip_high=clip_high, clip_c=args.clip_ratio_c)
                        loss = agg_loss(pg_mat, mask, args.loss_agg_mode, n_tokens, seqs_per_mini)
                        st["pg_loss"] += loss.item() / n_mini
                        if ref_logp is not None:
                            kl = agg_loss(kl_k3(logp, ref_logp[j]), mask, args.loss_agg_mode, n_tokens, seqs_per_mini)
                            loss = loss + args.kl_coef * kl
                            st["kl_loss"] += kl.item() / n_mini
                        loss.backward()
                        for key in ("clipped", "dual_clipped", "neg_log_ratio"):
                            st[key] += counts[key]
                        st["tokens"] += mask.sum().item()
                        if ratio_maxdev is None:
                            # 第一个 micro-batch 时参数还没动，ratio 理论上恰好是 1；偏离量就是两次前向的数值差
                            ratio_maxdev = (((logp.detach() - old_logp[j]).exp() - 1).abs() * mask).max().item()
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:  # 和 verl 一样：梯度出现 inf/nan 就跳过这次更新
                        st["skipped"] += 1
                        optimizer.zero_grad(set_to_none=True)
                    st["grad_norm"] += grad_norm.item() / n_mini

            with timed("update_weights", timing, device):
                sync_weights(policy, actor_rollout)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        val = {}
        if step % args.val_every == 0 or step == args.steps:
            with timed("testing", timing, device):
                val = validate(actor_rollout, tokenizer, val_items, args, device, step, out_dir)
            val_history.append({"step": step, **val})
        if args.save_every and step % args.save_every == 0 and step != args.steps:
            with timed("save_checkpoint", timing, device):
                save_model(actor_rollout, tokenizer, original_gen_config, out_dir / "checkpoints" / f"step_{step:04d}")

        lengths = torch.tensor([len(r) for r in responses], dtype=torch.float32)
        resp_tokens = lengths.sum().item()
        total_tokens = resp_tokens + sum(len(p) for p in prompts)
        grouped = rewards.view(-1, G)
        tok_adv_all = adv.cpu().repeat_interleave(lengths.long())  # 和 verl 一样按有效 token 统计优势
        metrics = {
            "step": step, "epoch": epoch,
            "critic/score/mean": rewards.mean().item(),
            "critic/score/max": rewards.max().item(),
            "critic/score/min": rewards.min().item(),
            "critic/advantages/mean": tok_adv_all.mean().item(),
            "critic/advantages/max": tok_adv_all.max().item(),
            "critic/advantages/min": tok_adv_all.min().item(),
            "groups/all_correct_frac": (grouped.min(dim=1).values == 1).float().mean().item(),
            "groups/all_wrong_frac": (grouped.max(dim=1).values == 0).float().mean().item(),
            "response_length/mean": lengths.mean().item(),
            "response_length/max": lengths.max().item(),
            "response_length/min": lengths.min().item(),
            "response_length/clip_ratio": 1 - sum(finished) / n,
            "prompt_length/mean": sum(len(p) for p in prompts) / n,
            "actor/pg_loss": st["pg_loss"],
            "actor/kl_loss": st["kl_loss"],
            "actor/entropy": ent_sum / tok_sum,
            "actor/pg_clipfrac": st["clipped"] / st["tokens"],
            "actor/pg_clipfrac_lower": st["dual_clipped"] / st["tokens"],
            "actor/ppo_kl": st["neg_log_ratio"] / st["tokens"],
            "actor/grad_norm": st["grad_norm"],
            "actor/lr": args.lr,
            "actor/skipped_updates": st["skipped"],
            "diag/ratio_maxdev_first_microbatch": ratio_maxdev,
            "diag/rollout_vs_train_logp_absdiff_mean": gap_mean,
            "diag/rollout_vs_train_logp_absdiff_max": gap_max,
            **{f"timing_s/{k}": v for k, v in timing.items()},
            "timing_per_token_ms/gen": timing["gen"] * 1000 / resp_tokens,
            "timing_per_token_ms/update_actor": timing["update_actor"] * 1000 / total_tokens,
            "perf/time_per_step": timing["step"],
            "perf/throughput": total_tokens / timing["step"],
            "perf/rollout_share": timing["gen"] / timing["step"],
            **val,
        }
        if device.type == "cuda":
            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()
        for g in range(min(2, len(picked))):
            i = g * G
            samples_file.write(json.dumps({"step": step, "question": train_items[picked[g]][2], "gold": golds[i],
                                           "reward": rewards[i].item(), "response": texts[i][:2000]},
                                          ensure_ascii=False) + "\n")
        samples_file.flush()

        for k, v in timing.items():
            timing_sums[k] = timing_sums.get(k, 0.0) + v
        recent_steps = (recent_steps + [timing["step"] + timing.get("testing", 0.0)])[-10:]
        eta_h = (args.steps - step) * sum(recent_steps) / len(recent_steps) / 3600
        line = (f"step {step}/{args.steps} | score {metrics['critic/score/mean']:.3f} | "
                f"kl {metrics['actor/kl_loss']:.5f} ent {metrics['actor/entropy']:.3f} | "
                f"len {metrics['response_length/mean']:.0f} clip {metrics['response_length/clip_ratio']:.2f} | "
                f"gen {timing['gen']:.1f}s upd {timing['update_actor']:.1f}s step {timing['step']:.1f}s | "
                f"eta {eta_h:.1f}h")
        if val:
            line += f" | val strict {val['val/acc_strict']:.3f} flex {val['val/acc_flexible']:.3f}"
        print(line)

    final_dir = out_dir / "final"
    save_model(actor_rollout, tokenizer, original_gen_config, final_dir)
    steps_done = args.steps
    mean_timing = {k: v / steps_done for k, v in timing_sums.items() if k not in ("testing", "save_checkpoint")}
    best = max(val_history, key=lambda v: v["val/acc_strict"])
    summary = {
        "steps": steps_done,
        "wall_clock_h": (time.perf_counter() - t_start) / 3600,
        "mean_timing_s": mean_timing,
        "rollout_share_mean": mean_timing["gen"] / mean_timing["step"],
        "val_history": val_history,
        "best_val_strict": {"step": best["step"], "acc_strict": best["val/acc_strict"]},
        "final_model": str(final_dir),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_file.close()
    samples_file.close()
    print(f"done | {summary['wall_clock_h']:.2f}h | rollout share {summary['rollout_share_mean']:.2f} | "
          f"val strict {val_history[0]['val/acc_strict']:.3f} -> {val_history[-1]['val/acc_strict']:.3f} | "
          f"final model: {final_dir}")


if __name__ == "__main__":
    main()
