"""P1 第 ④ 步：带 Python 工具的多轮 GRPO。

和第 ① 步的 GRPO 完全一样（组内优势、裁剪、k3 KL、token-mean），只有两处不同：
1. rollout 是多轮的：模型写代码 → 执行 → 看结果 → 再写，直到给出答案或到轮数上限（tool_rollout.py）；
2. response_mask 里工具返回的 token 是 0，所以 logprob 和 loss 都只算在模型生成的 token 上。

复用第 ① 步验证过的 grpo_core 和 train_grpo 的多数部件，这里只替换采样和拼 batch。

  python train_grpo_tool.py --model <底座或 SFT 模型> --data-dir <数据> --out-dir <目录>
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_core import agg_loss, grpo_advantages, kl_k3, policy_loss
from gsm8k import read_jsonl, score_strict
from rollout import forward_logprobs
from tool_rollout import collate_trajectories, rollout_with_tools, set_tool_generation
from train_grpo import prompt_batches, save_model, sync_weights, timed


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="底座，或第 ③ 步的 SFT 模型（带工具冷启动更稳）")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--train-batch-size", type=int, default=32, help="每步几道题（多轮采样慢，默认小一点）")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--mini-batch-size", type=int, default=8)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--gen-batch-size", type=int, default=128, help="每轮一次喂多少条活跃轨迹")
    ap.add_argument("--max-new-tokens", type=int, default=400, help="每一轮的生成上限")
    ap.add_argument("--max-prompt-len", type=int, default=400)
    ap.add_argument("--tool-timeout", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--clip-ratio", type=float, default=0.2)
    ap.add_argument("--clip-ratio-c", type=float, default=3.0)
    ap.add_argument("--kl-coef", type=float, default=0.001)
    ap.add_argument("--loss-agg-mode", default="token-mean", choices=["token-mean", "seq-mean-token-mean"])
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--debug-tool", action="store_true", help="冒烟测试用：极短轨迹也能跑通")
    return ap.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    args = parse_args()
    G = args.group_size
    if args.train_batch_size % args.mini_batch_size or (args.mini_batch_size * G) % args.micro_batch_size:
        raise SystemExit("batch 尺寸不整除，检查 train/mini/micro 与 group-size")
    n_mini = args.train_batch_size // args.mini_batch_size
    seqs_per_mini = args.mini_batch_size * G
    micro_per_mini = seqs_per_mini // args.micro_batch_size
    from pathlib import Path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    amp = torch.bfloat16 if device.type == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    policy = AutoModelForCausalLM.from_pretrained(args.model).to(device=device, dtype=torch.float32)
    original_gen = copy.deepcopy(policy.generation_config)
    ref = copy.deepcopy(policy).eval().requires_grad_(False) if args.kl_coef > 0 else None
    actor = copy.deepcopy(policy).to(dtype=amp or torch.float32).eval().requires_grad_(False)
    if not args.no_grad_ckpt:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.config.use_cache = False
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay, **({"fused": True} if device.type == "cuda" else {}))

    rows = read_jsonl(Path(args.data_dir) / "train.jsonl")
    items = [(r["extra_info"]["question"], r["reward_model"]["ground_truth"]) for r in rows
             if len(r["extra_info"]["question"]) < args.max_prompt_len * 4]
    (out_dir / "run_config.json").write_text(json.dumps({"args": vars(args), "n_train": len(items),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tool-grpo | train {len(items)} | {args.steps} steps x {args.train_batch_size} x {G} | "
          f"max_turns {args.max_turns} | kl {args.kl_coef}")

    metrics_file = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    batches = prompt_batches(len(items), args.train_batch_size, args.seed)
    t_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        timing: dict = {}
        _, picked = next(batches)
        questions = [items[i][0] for i in picked for _ in range(G)]
        golds = [items[i][1] for i in picked for _ in range(G)]
        n = len(questions)
        with timed("step", timing, device):
            with timed("gen", timing, device):
                gen_cfg = set_tool_generation(actor, tokenizer, greedy=False, max_new_tokens=args.max_new_tokens,
                                              temperature=args.temperature)
                trajs = rollout_with_tools(actor, tokenizer, questions, device=device, gen_config=gen_cfg,
                                           max_turns=args.max_turns, gen_batch_size=args.gen_batch_size,
                                           tool_timeout=args.tool_timeout)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            with timed("reward", timing, device):
                rewards = torch.tensor([score_strict(t.text, g) for t, g in zip(trajs, golds)])
            micro = [list(range(s, s + args.micro_batch_size)) for s in range(0, n, args.micro_batch_size)]
            mbs = [collate_trajectories([trajs[i] for i in idx], tokenizer.pad_token_id) for idx in micro]
            with timed("old_log_prob", timing, device), torch.no_grad():
                policy.eval()
                old_logp = [forward_logprobs(policy, b, device, amp)[0] for b in mbs]
            ref_logp = None
            if ref is not None:
                with timed("ref", timing, device), torch.no_grad():
                    ref_logp = [forward_logprobs(ref, b, device, amp)[0] for b in mbs]
            with timed("adv", timing, device):
                adv = grpo_advantages(rewards, G).to(device)
            with timed("update_actor", timing, device):
                policy.train()
                st = {"pg_loss": 0.0, "kl_loss": 0.0, "grad_norm": 0.0, "clipped": 0.0, "tokens": 0.0}
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
                                                     clip_high=args.clip_ratio, clip_c=args.clip_ratio_c)
                        loss = agg_loss(pg_mat, mask, args.loss_agg_mode, max(1.0, n_tokens), seqs_per_mini)
                        st["pg_loss"] += loss.item() / n_mini
                        if ref_logp is not None:
                            kl = agg_loss(kl_k3(logp, ref_logp[j]), mask, args.loss_agg_mode,
                                          max(1.0, n_tokens), seqs_per_mini)
                            loss = loss + args.kl_coef * kl
                            st["kl_loss"] += kl.item() / n_mini
                        loss.backward()
                        st["clipped"] += counts["clipped"]
                        st["tokens"] += mask.sum().item()
                    gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                    if torch.isfinite(gnorm):
                        optimizer.step()
                    st["grad_norm"] += gnorm.item() / n_mini
            with timed("update_weights", timing, device):
                sync_weights(policy, actor)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        grouped = rewards.view(-1, G)
        metrics = {
            "step": step, "critic/score/mean": rewards.mean().item(),
            "groups/all_wrong_frac": (grouped.max(dim=1).values == 0).float().mean().item(),
            "tool/call_rate": sum(t.tool_calls > 0 for t in trajs) / n,
            "tool/calls_mean": sum(t.tool_calls for t in trajs) / n,
            "tool/error_rate": sum(t.tool_errors for t in trajs) / max(1, sum(t.tool_calls for t in trajs)),
            "tool/turns_mean": sum(t.turns for t in trajs) / n,
            "tool/unfinished_rate": 1 - sum(t.finished for t in trajs) / n,
            "response_length/mean": sum(len(t.token_ids) for t in trajs) / n,
            "actor/pg_loss": st["pg_loss"], "actor/kl_loss": st["kl_loss"], "actor/grad_norm": st["grad_norm"],
            "actor/pg_clipfrac": st["clipped"] / max(1.0, st["tokens"]),
            **{f"timing_s/{k}": v for k, v in timing.items()},
            "perf/rollout_share": timing["gen"] / timing["step"],
        }
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()
        if args.save_every and step % args.save_every == 0:
            save_model(actor, tokenizer, original_gen, out_dir / "checkpoints" / f"step_{step:04d}")
        eta = (args.steps - step) * timing["step"] / 3600
        print(f"step {step}/{args.steps} | score {metrics['critic/score/mean']:.3f} | "
              f"tool_call {metrics['tool/call_rate']:.2f} err {metrics['tool/error_rate']:.2f} "
              f"turns {metrics['tool/turns_mean']:.2f} | len {metrics['response_length/mean']:.0f} | "
              f"gen {timing['gen']:.0f}s step {timing['step']:.0f}s | eta {eta:.1f}h")

    save_model(actor, tokenizer, original_gen, out_dir / "final")
    metrics_file.close()
    (out_dir / "run_summary.json").write_text(json.dumps({"steps": args.steps,
        "wall_clock_h": (time.perf_counter() - t_start) / 3600, "final_model": str(out_dir / "final"),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done | final model: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
