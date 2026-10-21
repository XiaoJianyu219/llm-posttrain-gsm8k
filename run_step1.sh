#!/usr/bin/env bash
# P1 第 ① 步 · 正式运行（带卡开机后，在 tmux 里跑）：
#   tmux new -s p1
#   cd /root/autodl-tmp/p1/llm-posttrain-gsm8k && bash run_step1.sh     # 额外参数传给 train_grpo.py，例如 --steps 100
#   按 Ctrl-b 再按 d 挂到后台；回来用 tmux attach -t p1
# 训练前基线（测试集）→ GRPO 训练（每 25 步测验证集）→ 训练后评测（测试集）→ 前后对比 → 打包结果（不含权重）
set -euo pipefail
cd "$(dirname "$0")"

ROOT=${ROOT:-/root/autodl-tmp/p1}
MODEL=${MODEL:-$ROOT/models/Qwen2.5-0.5B-Instruct}
DATA=${DATA:-$ROOT/data/gsm8k}
RUN=${RUN:-$ROOT/runs/step1_$(date +%m%d_%H%M)}
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
echo "运行目录：$RUN"

echo "[1/5] 检查环境"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'EOF'
import sys
import torch
import transformers
assert torch.cuda.is_available(), "没检测到 GPU：是不是还在无卡模式？"
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda,
      "| transformers", transformers.__version__, "|", torch.cuda.get_device_name(0),
      "| bf16", torch.cuda.is_bf16_supported())
EOF
test -f "$DATA/train.jsonl" || { echo "缺数据：先跑 setup_server.sh"; exit 1; }
test -f "$MODEL/config.json" || { echo "缺模型：先跑 setup_server.sh"; exit 1; }

echo "[2/5] 训练前基线：测试集 1319 题，贪心解码"
python eval_gsm8k.py --model "$MODEL" --data "$DATA/test.jsonl" --out-dir "$RUN/eval_base"

echo "[3/5] GRPO 训练"
python train_grpo.py --model "$MODEL" --data-dir "$DATA" --out-dir "$RUN/train" "$@"

echo "[4/5] 训练后评测：同一测试集、同一脚本"
python eval_gsm8k.py --model "$RUN/train/final" --data "$DATA/test.jsonl" --out-dir "$RUN/eval_final"

echo "[5/5] 前后对比 + 打包"
python compare_evals.py "$RUN/eval_base/samples.jsonl" "$RUN/eval_final/samples.jsonl" | tee "$RUN/compare.txt"
tar czf "$RUN.tgz" --exclude='*.safetensors' -C "$(dirname "$RUN")" "$(basename "$RUN")"
echo "完成。结果（不含权重）：$RUN.tgz"
