#!/usr/bin/env bash
# P1 第 ② 步 · 装 verl：独立 venv，绝不动第 ① 步在用的 /root/miniconda3。
# 用 verl v0.4.1（官方文档明确测过 vLLM 0.6.3），vLLM 0.6.3 锁定 torch 2.5.1+cu124，正好和服务器系统 torch 一致，
# 所以不会重新下载/覆盖 torch。走阿里云 pip 源（域内稳定），不用学术加速代理（之前它的隧道会断）。
# 只用 CPU 和网络，可以和第 ① 步训练同时跑。
#   bash setup_verl.sh
set -uo pipefail

ROOT=${ROOT:-/root/autodl-tmp/p1}
VERL_DIR=${VERL_DIR:-/root/autodl-tmp/verl}
VENV=${VENV:-/root/autodl-tmp/venvs/verl}
PY=/root/miniconda3/bin/python
PIP="$VENV/bin/pip"
MIRROR="https://mirrors.aliyun.com/pypi/simple"
export PIP_INDEX_URL=$MIRROR PIP_TRUSTED_HOST=mirrors.aliyun.com PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/setup_verl.log") 2>&1
echo "=== setup_verl $(date +%H:%M:%S) ==="

echo "[1/5] 建独立 venv"
[ -x "$PIP" ] || "$PY" -m venv "$VENV"
"$PIP" install -q -U pip wheel

echo "[2/5] 装 vLLM 0.6.3（它会带来 torch 2.5.1+cu124，与系统一致）"
# 用约束文件把 torch 钉死在系统同一版本，后面任何依赖都不许覆盖它
echo "torch==2.5.1" > "$ROOT/verl_constraints.txt"
"$PIP" install -c "$ROOT/verl_constraints.txt" "vllm==0.6.3"

echo "[3/5] 取 verl v0.4.1 源码"
[ -d "$VERL_DIR/.git" ] || git clone --branch v0.4.1 --depth 1 https://github.com/volcengine/verl.git "$VERL_DIR"
git -C "$VERL_DIR" describe --tags 2>/dev/null || git -C "$VERL_DIR" log --oneline -1

echo "[4/5] 装 verl（--no-deps，再按约束补运行时依赖，避免覆盖 torch/vllm）"
"$PIP" install --no-deps -e "$VERL_DIR"
"$PIP" install -c "$ROOT/verl_constraints.txt" \
  "ray[default]>=2.10" hydra-core tensordict codetiming dill pybind11 \
  pylatexenc "transformers>=4.45" accelerate datasets peft wandb tensorboard
# flash-attn 可选：装不上就跳过，verl 会退回 sdpa
"$PIP" install flash-attn==2.7.2.post1 --no-build-isolation \
  || echo "  flash-attn 跳过（可选，缺它用 sdpa，不影响正确性，只慢一点）"

echo "[5/5] 检查"
"$VENV/bin/python" - <<'EOF'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(),
      "| vllm", vllm.__version__, "| transformers", transformers.__version__)
import verl; print("verl ok:", verl.__file__)
assert torch.__version__.startswith("2.5.1"), "torch 被覆盖了，检查约束文件"
EOF
echo "verl 环境就绪：$VENV（verl 源码 $VERL_DIR）"
