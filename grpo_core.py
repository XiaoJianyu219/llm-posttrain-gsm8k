"""GRPO 的核心计算：纯张量函数，不依赖模型，可以单独测试。

形状约定：
- logp / old_logp / ref_logp：[B, T]，回答里每个 token 的 log π(token)
- mask：[B, T]，1 = 模型生成的有效 token（含结尾的 EOS），0 = padding
- adv：[B, T]，同一条回答的所有 token 共用一个优势

默认值与 verl 一致（2026-10-21 核对 verl main 分支源码），第 ② 步可以逐项对齐。
推导见 D:/LLM/0924/docs/learning/L0 第 5–7 节。
"""

from __future__ import annotations

import torch


def grpo_advantages(rewards: torch.Tensor, group_size: int, *, norm_by_std: bool = True,
                    eps: float = 1e-6) -> torch.Tensor:
    """组内优势。rewards 形状 [N]，同一道题的 G 条回答连续排列；返回形状 [N]。

    A_i = (R_i − 组内均值) / (组内标准差 + eps)。标准差用无偏估计（除以 G−1，torch.std 的默认），
    与 verl 相同。norm_by_std=False 时只减均值（Dr. GRPO 的做法）。组内奖励全相同时优势全为 0。
    """
    if group_size < 2:
        raise ValueError("GRPO 每组至少要 2 条回答")
    grouped = rewards.float().view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if norm_by_std:
        centered = centered / (grouped.std(dim=1, keepdim=True) + eps)
    return centered.reshape(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按 token mask 求均值；mask=0 的位置不贡献。"""
    return (values * mask).sum() / mask.sum().clamp_min(1)


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits [B, T, V] → targets 的 log 概率 [B, T]。

    转成 fp32 再算；logsumexp 逐条回答算，降低显存峰值（verl 也是这样做的）。
    """
    logits = logits.float()
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lse = torch.stack([torch.logsumexp(row, dim=-1) for row in logits])
    return picked - lse


@torch.no_grad()
def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """每个位置的熵 H = logsumexp(z) − Σ softmax(z)·z，只用于记录，不参与训练。"""
    out = []
    for row in logits.float():
        prob = torch.softmax(row, dim=-1)
        out.append(torch.logsumexp(row, dim=-1) - (prob * row).sum(dim=-1))
    return torch.stack(out)


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """k3 估计（verl 的 low_var_kl）：π_ref/π − log(π_ref/π) − 1，恒非负、方差小。截断范围与 verl 相同。"""
    log_ratio = (ref_logp - logp).clamp(-20.0, 20.0)
    return (log_ratio.exp() - log_ratio - 1.0).clamp(-10.0, 10.0)


def policy_loss(logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor, *,
                clip_low: float = 0.2, clip_high: float = 0.2, clip_c: float = 3.0):
    """裁剪目标的逐 token 损失（未聚合），以及用于记录的计数。

    ratio = π_θ / π_old，loss = −min(ratio·A, clip(ratio, 1−ε_low, 1+ε_high)·A)。
    A < 0 时再做一次 dual-clip：单个 token 的损失不超过 clip_c·|A|（verl 默认 3.0）。
    计数是 mask 内的 token 数之和，调用方汇总整个 mini-batch 后再除。
    """
    log_ratio = (logp - old_logp).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    loss_unclipped = -adv * ratio
    loss_clipped = -adv * ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    loss = torch.maximum(loss_unclipped, loss_clipped)
    cap = -adv * clip_c
    negative = adv < 0
    final = torch.where(negative, torch.minimum(loss, cap), loss)
    with torch.no_grad():
        counts = {
            "clipped": ((loss_clipped > loss_unclipped).float() * mask).sum().item(),
            "dual_clipped": (((loss > cap) & negative).float() * mask).sum().item(),
            "neg_log_ratio": (-log_ratio * mask).sum().item(),
        }
    return final, counts


def agg_loss(loss_mat: torch.Tensor, mask: torch.Tensor, mode: str, n_tokens: float, n_seqs: int) -> torch.Tensor:
    """逐 token 损失 → 标量。

    n_tokens / n_seqs 是整个 mini-batch 的总数，这样多个 micro-batch 的梯度累加起来，
    和一次算完整个 mini-batch 完全相同。
    token-mean：所有有效 token 一起平均（verl 默认，也是 DAPO 的做法）
    seq-mean-token-mean：先在每条回答内平均，再在回答之间平均（GRPO 论文原式）
    """
    if mode == "token-mean":
        return (loss_mat * mask).sum() / n_tokens
    if mode == "seq-mean-token-mean":
        per_seq = (loss_mat * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        return per_seq.sum() / n_seqs
    raise ValueError(f"不支持的 loss_agg_mode：{mode}")
