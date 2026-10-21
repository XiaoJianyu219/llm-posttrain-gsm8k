#!/usr/bin/env bash
# P1 第 ② 步 · 装 verl：独立 venv，绝不动第 ① 步在用的 /root/miniconda3。
# 服务器系统 torch 是 2.5.1+cu124；vLLM 0.6.4.post1 正好锁定 torch 2.5.1（0.6.3 锁的是 2.4.0，装不了）。
# venv 用 --system-site-packages 复用系统的 torch 和 CUDA 库，不重新下载 torch；只在 venv 里补 vllm 和 verl 依赖。
# transformers 在 venv 里钉到 vLLM 0.6.4 兼容的版本（系统的 4.57 太新）。走阿里云源，不用会断的学术代理。
# 只用 CPU 和网络，可与 GPU 上的训练同时跑。失败不致命（set 不加 -e），最后的检查会报清状态。
#   bash setup_verl.sh
set -uo pipefail

ROOT=${ROOT:-/root/autodl-tmp/p1}
VERL_DIR=${VERL_DIR:-/root/autodl-tmp/verl}
VENV=${VENV:-/root/autodl-tmp/venvs/verl}
PY=/root/miniconda3/bin/python
PIP="$VENV/bin/pip"
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple" PIP_TRUSTED_HOST=mirrors.aliyun.com
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/setup_verl.log") 2>&1
echo "=== setup_verl $(date +%H:%M:%S) ==="

echo "[1/5] 建独立 venv（--system-site-packages 复用系统 torch 2.5.1+cu124）"
[ -x "$PIP" ] || "$PY" -m venv --system-site-packages "$VENV"
"$PIP" install -q -U pip wheel
# 约束：torch 不许被换；transformers 钉到 vLLM 0.6.4 兼容版本（装进 venv，不影响系统）
printf "torch==2.5.1+cu124\ntorchvision\n" > "$ROOT/verl_constraints.txt"

echo "[2/5] 装 vLLM 0.6.4.post1（锁定 torch 2.5.1，已满足，不重装 torch）"
"$PIP" install -c "$ROOT/verl_constraints.txt" "vllm==0.6.4.post1" "transformers==4.47.1"

echo "[3/5] 取 verl v0.4.1 源码"
[ -d "$VERL_DIR/.git" ] || git clone --branch v0.4.1 --depth 1 https://github.com/volcengine/verl.git "$VERL_DIR"
git -C "$VERL_DIR" describe --tags 2>/dev/null || git -C "$VERL_DIR" log --oneline -1

echo "[4/5] 装 verl（--no-deps，再补运行时依赖，约束保护 torch/transformers）"
"$PIP" install --no-deps -e "$VERL_DIR"
"$PIP" install -c "$ROOT/verl_constraints.txt" \
  "ray[default]>=2.10" hydra-core tensordict codetiming dill pybind11 pylatexenc \
  accelerate datasets peft pyarrow pandas wandb tensorboard
# flash-attn 可选：装不上就跳过，verl 会退回 sdpa
"$PIP" install flash-attn==2.6.3 --no-build-isolation \
  || echo "  flash-attn 跳过（可选，缺它用 sdpa，不影响正确性，只慢一点）"

echo "[5/5] 检查"
"$VENV/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
assert torch.__version__.startswith("2.5.1"), "torch 被换了，检查约束文件"
import vllm, transformers, verl
print("vllm", vllm.__version__, "| transformers", transformers.__version__, "| verl", verl.__file__)
print("VERL_ENV_OK")
EOF
echo "verl 环境检查结束：$VENV（verl 源码 $VERL_DIR）"
