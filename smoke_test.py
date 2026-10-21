"""开训前的冒烟测试：只用 CPU，几分钟结束。setup_server.sh 最后一步会自动跑它。

  python smoke_test.py --model-dir <含 tokenizer 和 config.json 的模型目录> --raw-gsm8k <含 train.jsonl、test.jsonl 的目录>

分三部分：
1. 纯函数：判分（和按 verl 源码转写的严格判分逐例对照）、组内优势、裁剪损失、k3、loss 聚合、logprob、熵
2. 真实 tokenizer + 缩小的随机初始化 Qwen2 模型（带上真实的 generation_config.json，复现默认采样参数的坑）：
   - prompt 里有 chat template 和 #### 指令
   - 采样时没有 top-k / top-p / 重复惩罚：generate 返回的原始 logits 和处理后的 scores 完全一致
   - collate + forward_logprobs 算出的 logprob 和 generate 当时的 logits 一致（验证左填充、位置编码、切片）
3. 端到端：prepare_data → train_grpo 跑 2 步（组内交替奖励，保证有梯度）→ eval_gsm8k → compare_evals
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
from gsm8k import INSTRUCTION, build_row, encode_prompt, extract_gold, score_flexible, score_strict
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
    print("[1/3] 纯函数：通过")


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
    print("[2/3] tokenizer、采样配置、logprob 一致性：通过")


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
    print("[3/3] 端到端（准备数据 → 训练 2 步 → 评测 → 对比）：通过")


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
    finally:
        if args.keep:
            print(f"临时目录：{tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
