#!/usr/bin/env bash
# P1 第 ① 步 · 服务器准备：不需要 GPU，带卡、无卡模式都能跑（推荐带卡开机后直接跑，中间不关机，见 README）。
# 装依赖 → 下载并切分 GSM8K → 下载模型 → 冒烟测试。全部通过后接着跑 run_step1.sh。
#   cd /root/autodl-tmp/p1/llm-posttrain-gsm8k && bash setup_server.sh
set -euo pipefail
cd "$(dirname "$0")"

ROOT=${ROOT:-/root/autodl-tmp/p1}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_DIR=${MODEL_DIR:-$ROOT/models/Qwen2.5-0.5B-Instruct}
DATA=${DATA:-$ROOT/data/gsm8k}
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}            # 系统盘只有 30GB，缓存都放数据盘
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$ROOT/pip_cache}
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/setup_server.log") 2>&1

echo "[1/4] 安装依赖（torch 用镜像自带的，不重装）"
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
pip install -r requirements.txt

# AutoDL 的学术资源加速（GitHub、HuggingFace），只在下载数据和模型时打开
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; echo "已开启学术资源加速"; fi

echo "[2/4] 下载 GSM8K，切出 500 题验证集"
if [ -f "$DATA/test.jsonl" ]; then
  echo "已存在，跳过：$DATA"
else
  python prepare_data.py --out-dir "$DATA" || python prepare_data.py --out-dir "$DATA" --source hf
fi

echo "[3/4] 下载模型 $MODEL_ID -> $MODEL_DIR"
hf_download() {  # $1 = HF_ENDPOINT
  HF_ENDPOINT="$1" python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])' "$MODEL_ID" "$MODEL_DIR"
}
if [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "已存在，跳过"
else
  hf_download https://huggingface.co || hf_download https://hf-mirror.com \
    || { pip install modelscope && modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"; }
fi
ls -la "$MODEL_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "[4/4] 冒烟测试（CPU，几分钟）"
python smoke_test.py --model-dir "$MODEL_DIR" --raw-gsm8k "$DATA/raw"
echo "准备完成。接着跑：bash run_step1.sh"
