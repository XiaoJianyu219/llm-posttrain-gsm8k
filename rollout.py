"""采样、拼 batch、算 logprob。训练、验证、评测共用。"""

from __future__ import annotations

import torch
from transformers import GenerationConfig

from grpo_core import token_entropy, token_logprobs

EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")


def set_generation(model, tokenizer, *, greedy: bool, max_new_tokens: int, temperature: float = 1.0) -> None:
    """整份替换 model.generation_config。

    Qwen2.5-Instruct 自带的 generation_config.json 是 temperature=0.7、top_p=0.8、top_k=20、
    repetition_penalty=1.1。RL 要从策略本身的分布里采样（训练端算 logprob 用的是原始 softmax），
    所以显式关掉 top-k、top-p 和重复惩罚；贪心评测也要关掉重复惩罚。
    整份替换而不是传参覆盖，是为了不让模型自带的默认值混进来。
    """
    common = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=[tokenizer.convert_tokens_to_ids(t) for t in EOS_TOKENS],
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.0,
    )
    if greedy:
        model.generation_config = GenerationConfig(do_sample=False, **common)
    else:
        model.generation_config = GenerationConfig(
            do_sample=True, temperature=temperature, top_p=1.0, top_k=0, **common)


def left_pad(prompts: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, ids in enumerate(prompts):
        input_ids[row, width - len(ids):] = torch.tensor(ids)
        attention_mask[row, width - len(ids):] = 1
    return input_ids, attention_mask


@torch.no_grad()
def generate(model, tokenizer, prompts: list[list[int]], *, batch_size: int, device: torch.device):
    """左填充后分批生成，返回 (responses, finished)。

    responses[i] 是第 i 条的回答 token，截到第一个 EOS（含 EOS）；没遇到 EOS 说明被长度截断，finished[i]=False。
    先按 prompt 长度排序再分批，减少 padding；返回顺序与输入一致。
    """
    eos = set(model.generation_config.eos_token_id)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    responses: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        input_ids, attention_mask = left_pad([prompts[i] for i in chunk], tokenizer.pad_token_id)
        out = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        width = input_ids.shape[1]
        for row, i in enumerate(chunk):
            tokens = out[row, width:].tolist()
            cut = next((k for k, t in enumerate(tokens) if t in eos), None)
            responses[i] = tokens if cut is None else tokens[:cut + 1]
            finished[i] = cut is not None
    return responses, finished


def collate(prompts: list[list[int]], responses: list[list[int]], pad_id: int) -> dict:
    """左填充 prompt、右填充回答，拼成前向用的 batch。

    position_ids 从 attention_mask 推出来（cumsum − 1），和 generate 时一样；
    否则左填充会把每个 token 的位置整体往后挪，算出的 logprob 就和采样时对不上。
    """
    lp = max(len(p) for p in prompts)
    lr = max(len(r) for r in responses)
    b = len(prompts)
    input_ids = torch.full((b, lp + lr), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((b, lp + lr), dtype=torch.long)
    response_mask = torch.zeros((b, lr), dtype=torch.float32)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        input_ids[i, lp - len(p):lp] = torch.tensor(p)
        input_ids[i, lp:lp + len(r)] = torch.tensor(r)
        attention_mask[i, lp - len(p):lp + len(r)] = 1
        response_mask[i, :len(r)] = 1.0
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": input_ids[:, lp:].clone(),
        "response_mask": response_mask,
        "prompt_len": lp,
    }


def forward_logprobs(model, batch: dict, device: torch.device, autocast_dtype=None, *, with_entropy: bool = False):
    """对回答部分算 logprob（可选熵），形状 [b, lr]。

    只把回答位置的隐状态送进 lm_head：词表有 15 万，prompt 部分的 logits 用不上，算了白占显存。
    第 t 个位置的输出预测第 t+1 个 token，回答 token 在 [lp, lp+lr)，所以取隐状态 [lp−1, lp+lr−1)。
    autocast_dtype=None 时按模型自身精度算（bf16 采样副本、CPU 上的 fp32）。
    """
    lp = batch["prompt_len"]
    lr = batch["responses"].shape[1]
    with torch.autocast(device_type=device.type, dtype=autocast_dtype or torch.bfloat16,
                        enabled=autocast_dtype is not None):
        hidden = model.base_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
        ).last_hidden_state
        logits = model.get_output_embeddings()(hidden[:, lp - 1:lp + lr - 1])
    logp = token_logprobs(logits, batch["responses"].to(device))
    entropy = token_entropy(logits) if with_entropy else None
    return logp, entropy
