#!/usr/bin/env bash
# P1 第 ④ 步：带 Python 工具的多轮 GRPO，三条件对照。
#   bash run_step4.sh [起点模型]   起点默认底座；想做 SFT 冷启动就传第 ③ 步的 sft/final
# 三条件（都在测试集、同口径评测）：
#   A 不用工具      = eval_gsm8k 的训练后... 实际用底座贪心（= 第 ① 步 eval_base），这里不重复
#   B 用工具不训练  = 底座 + 工具评测
#   C 用工具并训练  = 本步训练后的模型 + 工具评测
# 另外和第 ① 步“用工具不训练 vs 单轮不训练”对照，看工具本身带来多少。
set -euo pipefail
cd "$(dirname "$0")"

ROOT=${ROOT:-/root/autodl-tmp/p1}
MODEL=${MODEL:-$ROOT/models/Qwen2.5-0.5B-Instruct}
START=${1:-$MODEL}                       # 训练起点；默认底座
DATA=${DATA:-$ROOT/data/gsm8k}
RUN=${RUN:-$ROOT/runs/step4_$(date +%m%d_%H%M)}
MAX_TURNS=${MAX_TURNS:-3}
GEN_BS=${GEN_BS:-128}
MAX_NEW=${MAX_NEW:-384}
STEPS=${STEPS:-100}
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
echo "运行目录：$RUN，起点 $START"
EVAL_ARGS="--max-turns $MAX_TURNS --gen-batch-size $GEN_BS --max-new-tokens $MAX_NEW"

echo "[1/4] 条件 B：用工具不训练（底座 + 工具，测试集）"
python eval_tool.py --model "$MODEL" --data "$DATA/test.jsonl" --out-dir "$RUN/eval_tool_base" $EVAL_ARGS

echo "[2/4] 带工具的多轮 GRPO"
python train_grpo_tool.py --model "$START" --data-dir "$DATA" --out-dir "$RUN/train" \
  --steps "$STEPS" --max-turns "$MAX_TURNS" --gen-batch-size "$GEN_BS" --max-new-tokens "$MAX_NEW"

echo "[3/4] 条件 C：用工具并训练（测试集）"
python eval_tool.py --model "$RUN/train/final" --data "$DATA/test.jsonl" --out-dir "$RUN/eval_tool_rl" $EVAL_ARGS

echo "[4/4] 打包（不含权重）"
tar czf "$RUN.tgz" --exclude='*.safetensors' -C "$(dirname "$RUN")" "$(basename "$RUN")"
echo "完成：$RUN.tgz"
echo "三条件对照：A 不用工具不训练 = <第①步 eval_base>；B 用工具不训练 = eval_tool_base；C 用工具并训练 = eval_tool_rl"
