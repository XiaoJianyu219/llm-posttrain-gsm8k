# llm-posttrain-gsm8k

小模型后训练实验：单张 RTX 4090，Qwen2.5-0.5B-Instruct + GSM8K，按顺序做四步。每一步都在**同一个留出测试集**上用**同一个评测脚本**量效果，结果可以直接放进一张表比较。

| 步骤 | 内容 | 状态 |
|---|---|---|
| ① | 自写 GRPO，训练前后在测试集上评测 | 代码完成，**还没在 GPU 上跑过** |
| ② | 用 verl 跑同一配置：逐项对齐，并把每步拆成 rollout / 参考模型 / 更新等段计时 | 未开始 |
| ③ | 同一底座做 SFT、DPO，和 GRPO 对比 | 未开始 |
| ④ | 给模型 Python 解释器工具，做多轮 GRPO | 未开始 |

---

## 第 ① 步怎么跑（AutoDL）

**租卡**：RTX 4090 × 1；基础镜像选 PyTorch ≥ 2.5、CUDA ≥ 12.4、Python 3.10–3.12；数据盘扩到 100 GB。

租到后**直接带卡开机，准备和训练一口气跑完，中间不要关机**：关机会释放 GPU，再开机时这台主机可能已经没有空闲卡。准备阶段十几分钟，带卡多花的钱很少。`setup_server.sh` 也能在无卡模式下跑，但按这个理由不推荐。

**1. 准备**

```bash
tmux new -s p1                                # 网页终端断开也不会中断
mkdir -p /root/autodl-tmp/p1 && cd /root/autodl-tmp/p1
source /etc/network_turbo                     # 学术资源加速，git clone GitHub 用
git clone https://github.com/XiaoJianyu219/llm-posttrain-gsm8k.git
unset http_proxy https_proxy
cd llm-posttrain-gsm8k && bash setup_server.sh
```

`setup_server.sh` 做四件事：装依赖（torch 用镜像自带的）→ 下载 GSM8K 并从训练集切出 500 题验证集 → 下载模型（HuggingFace，失败就换 hf-mirror，再失败换 ModelScope）→ 跑 `smoke_test.py`。最后打印 `ALL PASSED` 才算准备好。

仓库是私有的话，`git clone` 会要求登录：用户名填 GitHub 用户名，密码填一个对这个仓库有只读权限的 fine-grained token；或者在 GitHub 网页上 Download ZIP，再从 JupyterLab 上传。

**2. 训练**（在同一个 tmux 窗口里接着跑）

```bash
bash run_step1.sh
# Ctrl-b 再按 d 挂到后台；回来看进度：tmux attach -t p1
```

`run_step1.sh`：检查环境 → 训练前基线（测试集 1319 题，贪心）→ GRPO 训练（默认 150 步，每 25 步测一次验证集）→ 训练后评测（同一测试集、同一脚本）→ 前后对比 → 把结果打包成 `runs/step1_<时间>.tgz`（不含权重）。额外参数会传给 `train_grpo.py`，比如 `bash run_step1.sh --steps 100`。

**耗时（推算，没实测）**：每步 1–1.5 分钟，150 步约 2.5–4 小时，加上前后两次测试集评测各几分钟。前几步的日志里有 `eta`，以它为准。

**显存不够**时依次试：`--micro-batch-size 2`、`--gen-batch-size 128`。

---

## 文件

| 文件 | 作用 |
|---|---|
| `gsm8k.py` | 数据格式（与 verl 预处理产出的格式相同）、prompt、两种判分 |
| `grpo_core.py` | 纯张量函数：组内优势、裁剪损失（含 dual-clip）、k3 KL、loss 聚合、logprob、熵 |
| `rollout.py` | 生成配置、左填充分批生成、拼 batch、只对回答位置算 logprob |
| `prepare_data.py` | 下载 GSM8K，切验证集，写 jsonl 和 parquet（parquet 给第 ② 步的 verl 用） |
| `train_grpo.py` | GRPO 主循环，分段计时，写 `metrics.jsonl` |
| `eval_gsm8k.py` | 贪心评测：严格 / 宽松准确率、逐题结果；第 ②、③ 步的模型也用它测 |
| `compare_evals.py` | 两次评测的配对比较：McNemar 精确检验、“格式错 / 算错”拆分 |
| `smoke_test.py` | 上卡前在 CPU 上跑的测试（纯函数、采样配置、logprob 一致性、端到端 2 步） |
| `setup_server.sh` / `run_step1.sh` | 服务器上的准备和正式运行 |

---

## 实现要点

- **判分与 verl 严格模式一致**：只看回答最后 300 个字符，取最后一个 `#### 数字`，去逗号后按字符串比较。另有宽松判分（取最后一个数、按数值比较），只在评测时用来拆分“格式错”和“算错”。
- **采样参数整份替换**：Qwen2.5-Instruct 自带的 `generation_config.json` 是 temperature 0.7、top_p 0.8、top_k 20、repetition_penalty 1.1。RL 必须从策略本身的分布采样，所以采样时关掉 top-k、top-p 和重复惩罚，贪心评测也关掉重复惩罚。`smoke_test.py` 会检查 generate 的原始 logits 和处理后的 scores 完全一致。
- **两份权重**：训练端是 fp32 主权重，前向在 bf16 autocast 下算；采样用一份 bf16 副本（相当于 verl 里的推理引擎），每步更新后同步一次（`update_weights`）。
- **old logprob 在训练端重算**，参考模型是初始权重的冻结副本、和训练端同精度，所以第一个 micro-batch 的 ratio 理论上等于 1、第 1 步的 KL 等于 0。日志里的 `diag/*` 记录实际偏离多少，以及采样副本和训练端算出的 logprob 差多少（训推不一致的小测量）。
- **位置编码从 attention_mask 推出来**（左填充时和 generate 一致）；**只把回答位置送进 lm_head**（词表 15 万，prompt 部分的 logits 用不上）。
- **loss 聚合**默认 token-mean，分母取整个 mini-batch 的 token 数，梯度累积多个 micro-batch 后和一次算完相同；可切换成 GRPO 论文原式的 seq-mean-token-mean。
- **测试集只用两次**：训练中的监控只看从训练集切出的 500 题验证集，测试集只在训练前、训练后各测一次。

---

## 与 verl 的对应（第 ② 步照这些值设）

| 本仓库参数（默认值） | verl 配置项 |
|---|---|
| `--train-batch-size 64` | `data.train_batch_size` |
| `--group-size 8` | `actor_rollout_ref.rollout.n` |
| `--mini-batch-size 16` | `actor_rollout_ref.actor.ppo_mini_batch_size` |
| `--micro-batch-size 4` | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` |
| `--max-prompt-len 512` / `--max-response-len 512` | `data.max_prompt_length` / `data.max_response_length` |
| `--temperature 1.0`（top_p 1.0、top_k 关） | `actor_rollout_ref.rollout.temperature` 等 |
| `--lr 1e-6`、`--weight-decay 0.01`、`--grad-clip 1.0` | `actor.optim.lr`、`actor.optim.weight_decay`、`actor.grad_clip` |
| `--clip-ratio 0.2`、`--clip-ratio-c 3.0` | `actor.clip_ratio_low` / `clip_ratio_high`、`actor.clip_ratio_c` |
| `--kl-coef 0.001` | `actor.use_kl_loss=True`、`actor.kl_loss_coef=0.001`、`actor.kl_loss_type=low_var_kl` |
| `--loss-agg-mode token-mean` | `actor.loss_agg_mode` |
| `--no-std-norm`（默认不加） | `algorithm.norm_adv_by_std_in_grpo` |
| `--val-every 25` | `trainer.test_freq` |

verl 的默认值按 2026-10-21 的 main 分支核对；实际装的版本不同时，以那个版本的配置为准。

---

## 输出（`runs/step1_<时间>/`）

- `eval_base/`、`eval_final/`：训练前后的测试集评测（`summary.json` + 逐题 `samples.jsonl`）
- `compare.txt`：前后对比（两种口径的准确率、McNemar p 值、格式错 / 算错的题数）
- `train/metrics.jsonl`：每步一行。名字尽量和 verl 的日志一致：
  - `critic/score/mean`（平均奖励）、`critic/advantages/*`、`response_length/*`（含被截断比例 `clip_ratio`）
  - `actor/pg_loss`、`actor/kl_loss`、`actor/entropy`、`actor/pg_clipfrac`、`actor/ppo_kl`、`actor/grad_norm`
  - `timing_s/gen`、`reward`、`old_log_prob`、`ref`、`adv`、`update_actor`、`update_weights`、`step`；`perf/rollout_share`（rollout 占比）、`perf/throughput`
  - `groups/all_correct_frac`、`groups/all_wrong_frac`：组内全对 / 全错（优势为 0、没有梯度）的比例
  - `val/acc_strict`、`val/acc_flexible`、`val/format_rate`（验证步才有）
- `train/samples.jsonl`：每步两条采样；`train/val_stepXXXX.jsonl`：验证集逐题结果
- `train/run_config.json`、`train/run_summary.json`；`train/final/`、`train/checkpoints/`：权重（bf16，约 1 GB 一份，不打包）

`timing_s/step` 不含验证和存盘；verl 的 step 计时可能包含这两项，比较 rollout 占比时注意口径。

---

## 验证情况

- 本地只做了语法检查（`py_compile`、`bash -n`）。按计划不在本地装 torch、不在本地跑。
- `smoke_test.py` 设计成在服务器无卡模式下先跑一遍，通过后再用 GPU。**它还没运行过**，第一次跑如果报错，按报错改。
- GPU 上的完整训练还没跑过，上面的耗时都是推算。
