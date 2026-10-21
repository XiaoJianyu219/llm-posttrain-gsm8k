"""开训前的冒烟测试：只用 CPU，几分钟结束。setup_server.sh 最后一步会自动跑它。

  python smoke_test.py --model-dir <含 tokenizer 和 config.json 的模型目录> --raw-gsm8k <含 train.jsonl、test.jsonl 的目录>

分三部分：
1. 纯函数：判分（和按 verl 源码转写的严格判分逐例对照）、组内优势、裁剪损失、k3、loss 聚合、logprob、熵
2. 真实 tokenizer + 缩小的随机初始化 Qwen2 模型（带上真实的 generation_config.json，复现默认采样参数的坑）：
   - prompt 里有 chat template 和 #### 指令
   - 采样时没有 top-k / top-p / 重复惩罚：generate 返回的原始 logits 和处理后的 scores 完全一致
   - collate + forward_logprobs 算出的 logprob 和 generate 当时的 logits 一致（验证左填充、位置编码、切片）
3. 端到端：prepare_data → train_grpo 跑 2 步（组内交替奖励，保证有梯度）→ eval_gsm8k → compare_evals
4. 第 ③ 步：train_sft 跑 2 步；train_dpo 用人工构造的偏好对跑 4 步（第 1 步 margin 必须为 0）；make_dpo_pairs 能跑通
全部通过打印 ALL PASSED；任何一项不通过都会抛出 AssertionError，并写明是哪一项。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from grpo_core import agg_loss, grpo_advantages, kl_k3, policy_loss, token_entropy, token_logprobs
from gsm8k import (INSTRUCTION, build_row, encode_prompt, extract_gold, read_jsonl, score_flexible, score_strict,
                   write_jsonl)
from rollout import collate, forward_logprobs, left_pad, set_generation

HERE = Path(__file__).resolve().parent


def close(a, b, tol=1e-4) -> bool:
    return torch.allclose(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32), atol=tol)


def verl_strict_reference(solution: str, gold: str) -> float:
    """按 verl/utils/reward_score/gsm8k.py 严格模式的逻辑转写（2026-10-21 读源码），用来对照。"""
    found = re.findall(r"#### (\-?[0-9\.\,]+)", solution[-300:])
    if not found:
        return 0.0
    return 1.0 if found[-1].replace(",", "").replace("$", "") == gold else 0.0


def test_pure_functions() -> None:
    cases = [
        ("Janet has 16 eggs, 16-3-4=9, 9*2=18. #### 18", "18", 1.0),
        ("18", "18", 0.0),
        ("So the answer is #### 18.", "18", 0.0),
        ("####18", "18", 0.0),
        ("#### 9 ... wait, #### 18", "18", 1.0),
        ("#### 18 " + "x" * 400, "18", 0.0),
        ("#### 1,000", "1000", 1.0),
        ("#### -5", "-5", 1.0),
        ("#### 18.0", "18", 0.0),
    ]
    for text, gold, expected in cases:
        assert score_strict(text, gold) == expected == verl_strict_reference(text, gold), f"判分不一致：{text[:40]!r}"
    assert extract_gold("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 1,234") == "1234", "extract_gold"
    assert score_flexible("The answer is $18.00.", "18") and score_flexible("#### 18.", "18"), "flexible 应判对"
    assert not score_flexible("no number here", "18"), "flexible 应判错"

    adv = grpo_advantages(torch.tensor([1.0, 0, 0, 1]), 4)
    assert close(adv, [0.8660, -0.8660, -0.8660, 0.8660]), f"优势 [1,0,0,1]：{adv}"
    assert close(grpo_advantages(torch.tensor([1.0, 1, 1, 0]), 4), [0.5, 0.5, 0.5, -1.5]), "优势 [1,1,1,0]"
    assert close(grpo_advantages(torch.tensor([1.0, 1, 1, 1]), 4), [0, 0, 0, 0]), "全对的组优势应为 0"
    adv8 = grpo_advantages(torch.tensor([1.0] + [0.0] * 7), 8)
    assert close(adv8[0], 2.4749) and close(adv8[1:], [-0.3536] * 7), f"G=8 一个对：{adv8}"
    assert close(grpo_advantages(torch.tensor([1.0, 0, 0, 1]), 4, norm_by_std=False), [0.5, -0.5, -0.5, 0.5]), "Dr. GRPO"

    lp = torch.randn(3, 5)
    assert close(kl_k3(lp, lp), torch.zeros(3, 5)), "相同分布的 k3 应为 0"
    assert (kl_k3(lp, torch.randn(3, 5)) >= 0).all(), "k3 应非负"
    assert kl_k3(torch.zeros(1), torch.full((1,), 50.0)).item() <= 10.0, "k3 应截断在 10"

    mask = torch.ones(1, 1)
    old = torch.zeros(1, 1)
    loss, c = policy_loss(old.clone(), old, torch.full((1, 1), 2.0), mask)
    assert close(loss, -2.0) and c["clipped"] == 0, "ratio=1 时损失应为 −A"
    loss, c = policy_loss(torch.full((1, 1), math.log(1.5)), old, torch.ones(1, 1), mask)
    assert close(loss, -1.2) and c["clipped"] == 1, "A>0、ratio=1.5 应裁到 1.2"
    loss, _ = policy_loss(torch.full((1, 1), math.log(0.5)), old, -torch.ones(1, 1), mask)
    assert close(loss, 0.8), "A<0、ratio=0.5 应裁到 0.8"
    loss, c = policy_loss(torch.full((1, 1), math.log(5.0)), old, -torch.ones(1, 1), mask)
    assert close(loss, 3.0) and c["dual_clipped"] == 1, "A<0、ratio=5 应被 dual-clip 限在 3"

    loss_mat = torch.tensor([[1.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert close(agg_loss(loss_mat, m, "token-mean", m.sum().item(), 2), 5 / 3), "token-mean"
    assert close(agg_loss(loss_mat, m, "seq-mean-token-mean", m.sum().item(), 2), 2.0), "seq-mean-token-mean"
    split = sum(agg_loss(loss_mat[i:i + 1], m[i:i + 1], "token-mean", m.sum().item(), 2) for i in range(2))
    assert close(split, 5 / 3), "拆成 micro-batch 累加应与整批相同"

    logits = torch.randn(2, 4, 11)
    targets = torch.randint(0, 11, (2, 4))
    expected = torch.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert close(token_logprobs(logits, targets), expected), "token_logprobs"
    p = torch.softmax(logits, -1)
    assert close(token_entropy(logits), -(p * p.log()).sum(-1)), "token_entropy"
    print("[1/5] 纯函数：通过")


def make_tiny_model(model_dir: Path, out: Path):
    """缩小版 Qwen2：真实 tokenizer 和词表大小，2 层、隐藏维 64；带上真实的 generation_config.json。"""
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.hidden_size, cfg.intermediate_size = 64, 128
    cfg.num_hidden_layers, cfg.max_window_layers = 2, 2
    if getattr(cfg, "layer_types", None):  # 新版 transformers 按层记录注意力类型，重新加载时会和层数对照
        cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    if hasattr(cfg, "head_dim"):
        cfg.head_dim = 16
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).float()  # from_config 可能按 config 里的 bf16 建模型
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(out)
    shutil.copy(model_dir / "generation_config.json", out / "generation_config.json")
    return out


def test_tokenizer_and_generation(model_dir: Path, tiny_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    rows = [
        build_row("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                  "How many clips did Natalia sell altogether in April and May?", "48/2 = 24\n48+24 = 72\n#### 72",
                  "test", 0),
        build_row("What is 2 + 3?", "2 + 3 = 5\n#### 5", "test", 1),
    ]
    prompts = [encode_prompt(tok, r) for r in rows]
    text = tok.decode(prompts[0])
    assert "<|im_start|>user" in text and INSTRUCTION in text, "prompt 里缺 chat template 或指令"
    assert text.endswith("<|im_start|>assistant\n"), f"prompt 结尾应是生成提示：{text[-40:]!r}"

    model = AutoModelForCausalLM.from_pretrained(tiny_dir).float().eval()
    raw = model.generation_config
    assert raw.top_p == 0.8 and raw.repetition_penalty == 1.1, "没复现出 Qwen 默认采样参数（测试本身的前提）"
    set_generation(model, tok, greedy=False, max_new_tokens=8)
    cfg = model.generation_config
    assert cfg.top_p == 1.0 and cfg.top_k == 0 and cfg.repetition_penalty == 1.0 and cfg.temperature == 1.0, "采样配置"

    input_ids, attention_mask = left_pad(prompts, tok.pad_token_id)
    assert input_ids.shape[0] == 2 and attention_mask[1].sum() < attention_mask.shape[1], "第二条应被左填充"
    torch.manual_seed(1)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, output_scores=True,
                             output_logits=True, return_dict_in_generate=True)
    for t, (scores, logits) in enumerate(zip(out.scores, out.logits)):
        assert torch.equal(scores.float(), logits.float()), f"第 {t} 步采样分布被改过（top-k/top-p/重复惩罚没关干净）"

    width = input_ids.shape[1]
    eos = set(cfg.eos_token_id)
    responses = []
    for row in range(2):
        tokens = out.sequences[row, width:].tolist()
        cut = next((k for k, t in enumerate(tokens) if t in eos), None)
        responses.append(tokens if cut is None else tokens[:cut + 1])
    batch = collate(prompts, responses, tok.pad_token_id)
    with torch.no_grad():
        logp, ent = forward_logprobs(model, batch, torch.device("cpu"), None, with_entropy=True)
    for row, resp in enumerate(responses):
        for t, token in enumerate(resp):
            gen_lp = torch.log_softmax(out.logits[t][row].float(), -1)[token]
            assert abs(gen_lp.item() - logp[row, t].item()) < 1e-4, f"第 {row} 条第 {t} 个 token 的 logprob 对不上"
    assert torch.isfinite(ent).all(), "熵出现 nan/inf"

    set_generation(model, tok, greedy=True, max_new_tokens=8)
    assert not model.generation_config.do_sample and model.generation_config.repetition_penalty == 1.0, "贪心配置"
    print("[2/5] tokenizer、采样配置、logprob 一致性：通过")


def run(args: list, label: str) -> None:
    result = subprocess.run([sys.executable, *map(str, args)], cwd=HERE, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-3000:], sep="\n")
        raise AssertionError(f"{label} 运行失败（退出码 {result.returncode}）")


def test_end_to_end(raw_gsm8k: Path, tiny_dir: Path, tmp: Path) -> None:
    data = tmp / "data"
    run(["prepare_data.py", "--source", "local", "--raw-dir", raw_gsm8k, "--out-dir", data, "--val-size", "20"],
        "prepare_data")
    info = json.loads((data / "split_info.json").read_text(encoding="utf-8"))
    assert info["counts"]["val"] == 20 and info["counts"]["test"] == 1319, f"数据切分：{info['counts']}"

    run_dir = tmp / "run"
    run(["train_grpo.py", "--model", tiny_dir, "--data-dir", data, "--out-dir", run_dir, "--steps", "2",
         "--train-batch-size", "2", "--group-size", "4", "--mini-batch-size", "1", "--micro-batch-size", "2",
         "--gen-batch-size", "8", "--max-response-len", "16", "--val-every", "2", "--val-limit", "4",
         "--val-batch-size", "4", "--save-every", "1", "--lr", "1e-3", "--debug-reward", "alternate",
         "--device", "cpu"], "train_grpo")
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["step"] for x in lines] == [0, 1, 2], "metrics.jsonl 应有第 0、1、2 步"
    required = ["timing_s/gen", "timing_s/reward", "timing_s/old_log_prob", "timing_s/ref", "timing_s/adv",
                "timing_s/update_actor", "timing_s/update_weights", "timing_s/step", "critic/score/mean",
                "critic/advantages/mean", "response_length/clip_ratio", "actor/pg_loss", "actor/kl_loss",
                "actor/entropy", "actor/pg_clipfrac", "actor/grad_norm", "perf/rollout_share",
                "diag/ratio_maxdev_first_microbatch", "diag/rollout_vs_train_logp_absdiff_mean"]
    missing = [k for k in required if k not in lines[1]]
    assert not missing, f"缺指标：{missing}"
    assert "val/acc_strict" in lines[2], "最后一步应有验证集结果"
    assert lines[1]["diag/ratio_maxdev_first_microbatch"] < 1e-4, "CPU fp32 下第一个 micro-batch 的 ratio 应为 1"
    assert lines[2]["actor/kl_loss"] > 0, "参数更新后 KL 应大于 0"
    assert (run_dir / "checkpoints" / "step_0001").exists() and (run_dir / "final").exists(), "检查点没存下来"
    saved_cfg = json.loads((run_dir / "final" / "generation_config.json").read_text(encoding="utf-8"))
    assert saved_cfg.get("top_p") == 0.8, "保存时应恢复模型原来的 generation_config"
    before = AutoModelForCausalLM.from_pretrained(tiny_dir).state_dict()
    after = AutoModelForCausalLM.from_pretrained(run_dir / "final").state_dict()
    assert max((before[k].float() - after[k].float()).abs().max().item() for k in before) > 0, "参数没被更新"

    for name, model in (("eval_base", tiny_dir), ("eval_final", run_dir / "final")):
        run(["eval_gsm8k.py", "--model", model, "--data", data / "test.jsonl", "--out-dir", tmp / name,
             "--limit", "6", "--batch-size", "3", "--max-new-tokens", "16", "--device", "cpu"], name)
        summary = json.loads((tmp / name / "summary.json").read_text(encoding="utf-8"))
        assert summary["n"] == 6 and "acc_flexible" in summary, f"{name} summary"
    run(["compare_evals.py", tmp / "eval_base" / "samples.jsonl", tmp / "eval_final" / "samples.jsonl"],
        "compare_evals")
    run(["summarize_evals.py", f"base={tmp / 'eval_base'}", f"grpo={tmp / 'eval_final'}"], "summarize_evals")
    print("[3/5] 端到端（准备数据 → 训练 2 步 → 评测 → 对比）：通过")


def test_step3(tiny_dir: Path, tmp: Path) -> None:
    """第 ③ 步：SFT 跑 2 步；DPO 用人工构造的偏好对跑 4 步；造偏好对的采样脚本能跑通。"""
    data = tmp / "data"
    run(["train_sft.py", "--model", tiny_dir, "--data-dir", data, "--out-dir", tmp / "sft", "--epochs", "1",
         "--limit", "8", "--batch-size", "4", "--micro-batch-size", "2", "--max-response-len", "400",
         "--val-batch-size", "10", "--lr", "1e-3", "--device", "cpu"], "train_sft")
    lines = [json.loads(x) for x in (tmp / "sft" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    losses = [x["train/loss"] for x in lines if "train/loss" in x]
    assert len(losses) == 2 and all(math.isfinite(v) and v > 0 for v in losses), f"SFT loss：{losses}"
    assert (tmp / "sft" / "final").exists(), "SFT 没存下模型"

    tok = AutoTokenizer.from_pretrained(tiny_dir)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    pairs = []
    for row in read_jsonl(data / "train.jsonl")[:4]:
        gold = row["reward_model"]["ground_truth"]
        pairs.append({"index": row["extra_info"]["index"], "gold": gold, "prompt_ids": encode_prompt(tok, row),
                      "chosen_ids": tok(f"#### {gold}", add_special_tokens=False)["input_ids"] + [im_end],
                      "rejected_ids": tok("no idea", add_special_tokens=False)["input_ids"] + [im_end],
                      "chosen": f"#### {gold}", "rejected": "no idea"})
    write_jsonl(tmp / "pairs.jsonl", pairs)
    run(["train_dpo.py", "--model", tiny_dir, "--pairs", tmp / "pairs.jsonl", "--data-dir", data, "--out-dir",
         tmp / "dpo", "--epochs", "2", "--batch-size", "2", "--micro-batch-size", "1", "--lr", "1e-3",
         "--max-response-len", "16", "--val-batch-size", "10", "--device", "cpu"], "train_dpo")
    steps = [json.loads(x) for x in (tmp / "dpo" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    steps = [x for x in steps if "train/margin" in x]
    assert len(steps) == 4, f"DPO 应有 4 步，实际 {len(steps)}"
    assert abs(steps[0]["train/margin"]) < 1e-4, f"第 1 步更新前策略等于参考模型，margin 应为 0：{steps[0]}"
    assert steps[-1]["train/margin"] > steps[0]["train/margin"], "DPO 几步后 margin 应该变大"
    assert (tmp / "dpo" / "final").exists(), "DPO 没存下模型"

    run(["make_dpo_pairs.py", "--model", tiny_dir, "--data-dir", data, "--out", tmp / "pairs_gen.jsonl",
         "--limit", "2", "--samples-per-prompt", "2", "--max-new-tokens", "8", "--device", "cpu"], "make_dpo_pairs")
    assert (tmp / "pairs_gen_stats.json").exists(), "造偏好对没写出统计"
    print("[4/5] 第 ③ 步（SFT 2 步、DPO 4 步、造偏好对）：通过")


def test_tool(tiny_dir: Path, tmp: Path) -> None:
    """第 ④ 步：工具执行、轨迹 mask 对齐、多轮采样跑通、带工具 GRPO 跑 2 步。"""
    import torch
    from tool_rollout import Trajectory, collate_trajectories, rollout_with_tools, run_python, set_tool_generation

    assert run_python("print(6*7)") == ("42", False), "工具应执行代码并返回 stdout"
    assert run_python("print(sum(range(5)))") == ("10", False), "工具 sum"
    out, err = run_python("1/0")
    assert err and "ZeroDivisionError" in out, "代码抛异常应回灌 traceback 并标记出错，不让采样挂掉"
    out, err = run_python("while True: pass", timeout=1.0)
    assert err and "timed out" in out, "死循环应超时并标记出错"

    # 手工轨迹：模型 2 token、工具 2 token、模型 1 token → response_mask 必须是 [1,1,0,0,1]
    t = Trajectory(prompt_ids=[10, 11], token_ids=[20, 21, 30, 31, 22], mask=[1, 1, 0, 0, 1])
    batch = collate_trajectories([t], pad_id=0)
    assert batch["response_mask"][0].tolist() == [1, 1, 0, 0, 1], "工具 token 的 mask 必须是 0"
    assert batch["response_mask"].sum().item() == 3, "只有 3 个模型生成 token 计入"
    assert batch["prompt_len"] == 2 and batch["responses"].shape[1] == 5, "collate 形状"

    tok = AutoTokenizer.from_pretrained(tiny_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir).float().eval()
    gen = set_tool_generation(model, tok, greedy=True, max_new_tokens=8)
    trajs = rollout_with_tools(model, tok, ["What is 2+3?"], device=torch.device("cpu"), gen_config=gen,
                               max_turns=2, tool_timeout=3.0)
    tr = trajs[0]
    assert tr.prompt_len > 0 and len(tr.token_ids) == len(tr.mask), "轨迹 mask 必须和 token 等长"
    assert all(m in (0, 1) for m in tr.mask) and set(tr.mask) <= {0, 1}, "mask 只能是 0/1"

    run(["train_grpo_tool.py", "--model", tiny_dir, "--data-dir", tmp / "data", "--out-dir", tmp / "tool",
         "--steps", "2", "--train-batch-size", "2", "--group-size", "4", "--mini-batch-size", "1",
         "--micro-batch-size", "2", "--max-turns", "2", "--max-new-tokens", "8", "--device", "cpu"],
        "train_grpo_tool")
    lines = [json.loads(x) for x in (tmp / "tool" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and "tool/call_rate" in lines[0] and "timing_s/gen" in lines[0], "工具 GRPO 指标缺失"
    assert (tmp / "tool" / "final").exists(), "工具 GRPO 没存下模型"
    print("[5/5] 第 ④ 步（工具执行、轨迹 mask、多轮采样、带工具 GRPO 2 步）：通过")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--raw-gsm8k", type=Path, required=True)
    ap.add_argument("--keep", action="store_true", help="保留临时目录，方便排查")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="p1_smoke_"))
    try:
        test_pure_functions()
        tiny_dir = make_tiny_model(args.model_dir, tmp / "tiny")
        test_tokenizer_and_generation(args.model_dir, tiny_dir)
        test_end_to_end(args.raw_gsm8k, tiny_dir, tmp)
        test_step3(tiny_dir, tmp)
        test_tool(tiny_dir, tmp)
    finally:
        if args.keep:
            print(f"临时目录：{tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
