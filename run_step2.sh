#!/usr/bin/env bash
# P1 第 ② 步 · 用 verl 跑同一配置的 GSM8K GRPO，和第 ① 步自写实现对齐 + 看 verl 的耗时分解。
# 配置照 verl v0.4.1 的 examples/grpo_trainer/run_qwen2-7b.sh，改成单卡 0.5B，并对齐本仓库第 ① 步的超参：
#   adv=grpo、rollout.n=8、lr 1e-6、kl_loss_coef 0.001、low_var_kl、clip 0.2、max_prompt/response 512。
# 数据直接用 prepare_data 写的 parquet（本仓库数据格式就是照 verl 建的）。verl 自带 gsm8k 严格判分，和本仓库一致。
#   bash run_step2.sh
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/p1}
MODEL=${MODEL:-$ROOT/models/Qwen2.5-0.5B-Instruct}
DATA=${DATA:-$ROOT/data/gsm8k}
VERL_PY=${VERL_PY:-/root/autodl-tmp/venvs/verl/bin/python}
RUN=${RUN:-$ROOT/runs/step2_verl_$(date +%m%d_%H%M)}
mkdir -p "$RUN"
export VLLM_USE_V1=0                       # vllm 0.6.4 用旧引擎路径，和 verl 0.4.1 搭
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
exec > >(tee -a "$RUN/run.log") 2>&1
echo "verl GRPO 单卡 0.5B，运行目录 $RUN"
test -f "$DATA/train.parquet" || { echo "缺 parquet，先在主环境跑 prepare_data（已装 pandas/pyarrow）"; exit 1; }

"$VERL_PY" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/test.parquet" \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=p1_verl_gsm8k \
  trainer.experiment_name=qwen0.5b_grpo \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.total_epochs=1
echo "verl GRPO 结束：$RUN"
