#!/usr/bin/env bash
# P1 第 ② 步 · 装 verl：独立环境，不动第 ① 步用的 /root/miniconda3。
# 按 verl 官方文档（2026-10-22 查）的 uv + 锁文件方式：vLLM >= 0.18、CUDA >= 12.8、Python 3.12。
# 只用 CPU 和网络，可以在第 ① 步训练的同时跑。
#   bash setup_verl.sh
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/p1}
VERL_DIR=${VERL_DIR:-/root/autodl-tmp/verl}
UV_VENV=/root/autodl-tmp/venvs/uvtool
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/autodl-tmp/uv_cache}   # 系统盘只有 30GB，缓存放数据盘
export PATH=/root/miniconda3/bin:$PATH                          # 让 uv 找到现成的 Python 3.12
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/setup_verl.log") 2>&1

echo "[1/4] 装 uv（放在单独的 venv 里）"
if [ ! -x "$UV_VENV/bin/uv" ]; then
  python -m venv "$UV_VENV" && "$UV_VENV/bin/pip" install -q uv
fi
UV="$UV_VENV/bin/uv"
"$UV" --version

if [ -f /etc/network_turbo ]; then source /etc/network_turbo; echo "已开启学术资源加速"; fi

echo "[2/4] 取 verl 源码"
[ -d "$VERL_DIR/.git" ] || git clone --depth 1 https://github.com/verl-project/verl.git "$VERL_DIR"
git -C "$VERL_DIR" log --oneline -1

echo "[3/4] 按锁文件建环境（vllm + fsdp）"
cd "$VERL_DIR"
"$UV" sync --frozen --all-packages --extra vllm --extra fsdp

echo "[4/4] 检查"
.venv/bin/python -c "import torch, vllm, verl, transformers; print('torch', torch.__version__, torch.version.cuda, '| vllm', vllm.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
echo "verl 环境就绪：$VERL_DIR/.venv"
