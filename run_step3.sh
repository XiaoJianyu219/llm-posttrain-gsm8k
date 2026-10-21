#!/usr/bin/env bash
# P1 第 ③ 步：SFT、DPO，和第 ① 步的 GRPO 在同一个测试集、同一个评测脚本上比较。
#   bash run_step3.sh <第 ① 步的运行目录>        例如 /root/autodl-tmp/p1/runs/step1_1022_0100
# 造偏好对默认用 hf 采样；有装了 vllm 的 Python 时设 PAIRS_PY=<那个 python> BACKEND=vllm，快很多。
set -euo pipefail
cd "$(dirname "$0")"

ROOT=${ROOT:-/root/autodl-tmp/p1}
MODEL=${MODEL:-$ROOT/models/Qwen2.5-0.5B-Instruct}
DATA=${DATA:-$ROOT/data/gsm8k}
STEP1=${1:?用法：bash run_step3.sh <第 ① 步的运行目录>}
RUN=${RUN:-$ROOT/runs/step3_$(date +%m%d_%H%M)}
BACKEND=${BACKEND:-hf}
PAIRS_PY=${PAIRS_PY:-python}
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
echo "运行目录：$RUN"

echo "[1/5] SFT（标准解答，2 个 epoch）"
python train_sft.py --model "$MODEL" --data-dir "$DATA" --out-dir "$RUN/sft"
python eval_gsm8k.py --model "$RUN/sft/final" --data "$DATA/test.jsonl" --out-dir "$RUN/eval_sft"

# 底座几乎不写 ####，严格判分配不出偏好对，所以按数值对错（flexible）配对；评测仍同时报严格 / 宽松
echo "[2/5] 用底座采样，按数值对错造 DPO 偏好对（$BACKEND）"
"$PAIRS_PY" make_dpo_pairs.py --model "$MODEL" --data-dir "$DATA" --out "$RUN/dpo_pairs.jsonl" \
  --backend "$BACKEND" --scorer flexible

echo "[3/5] DPO"
python train_dpo.py --model "$MODEL" --pairs "$RUN/dpo_pairs.jsonl" --data-dir "$DATA" --out-dir "$RUN/dpo"
python eval_gsm8k.py --model "$RUN/dpo/final" --data "$DATA/test.jsonl" --out-dir "$RUN/eval_dpo"

echo "[4/5] 底座 / SFT / DPO / GRPO 放一张表"
python summarize_evals.py base="$STEP1/eval_base" sft="$RUN/eval_sft" dpo="$RUN/eval_dpo" \
  grpo="$STEP1/eval_final" | tee "$RUN/summary.txt"

echo "[5/5] 打包（不含权重）"
tar czf "$RUN.tgz" --exclude='*.safetensors' -C "$(dirname "$RUN")" "$(basename "$RUN")"
echo "完成：$RUN.tgz"
