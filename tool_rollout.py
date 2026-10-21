"""带 Python 工具的多轮轨迹采样（第 ④ 步用）。

模型可以写 ```python ... ``` 代码块；代码在隔离子进程里执行，输出作为一条 user 消息拼回去，
模型看到结果后继续推理，最后给出 "#### 答案"。关键：工具返回的 token 不计入 loss。

轨迹的 token 序列通过**增量分词**拼出来，同时给出 response_mask（1=模型生成，0=工具返回 / 模板）：
每一轮把当前 messages 用 chat template 渲染，和上一轮的渲染结果比较，新增的部分就是这一轮的 token；
assistant 轮标 1，tool 轮标 0。这样重算 logprob 时和采样时的 token 完全对齐。
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import GenerationConfig

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
TOOL_INSTRUCTION = (
    "You may use Python to help. Write code in a ```python ... ``` block; it will be executed and the "
    "result returned to you as `stdout`. Use print() to see values. After reasoning, give the final "
    'numeric answer after "####". Question: '
)


def build_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": TOOL_INSTRUCTION + question}]


def run_python(code: str, timeout: float = 5.0) -> tuple[str, bool]:
    """在隔离子进程里跑一段代码，返回 (回灌给模型的文本, 是否出错)。
    -I -S 隔离，不继承环境、site 和当前目录的模块。代码自身抛异常时把 traceback 回灌（真实解释器就是这样），
    并把 is_error 置真，供统计工具失败率。"""
    is_error = False
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "snippet.py"
        f.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", "-S", str(f)], capture_output=True, text=True,
                               timeout=timeout, cwd=d)
            out = (p.stdout or "") + (("\n" + p.stderr) if p.returncode and p.stderr else "")
            is_error = p.returncode != 0
        except subprocess.TimeoutExpired:
            out, is_error = f"[tool error] execution timed out after {timeout:g}s", True
        except Exception as exc:  # noqa: BLE001 - 工具错误要回灌给模型，不能让采样挂掉
            out, is_error = f"[tool error] {type(exc).__name__}: {exc}", True
    out = out.strip()
    if len(out) > 600:
        out = out[:600] + " …[truncated]"
    return (out or "[no output]"), is_error


@dataclass
class Trajectory:
    prompt_ids: list = field(default_factory=list)  # 第一段 prompt（含工具说明）的 token
    token_ids: list = field(default_factory=list)   # prompt 之后的所有 token（response 部分）
    mask: list = field(default_factory=list)        # 与 token_ids 等长，1=模型生成，0=工具返回
    turns: int = 0                        # 模型生成了几轮
    tool_calls: int = 0                   # 成功执行了几次代码
    tool_errors: int = 0                  # 代码报错/超时几次
    finished: bool = False                # 是否自然结束（给出了答案或 EOS），而非被轮数/长度截断
    text: str = ""                        # 整条 assistant 可读文本（拼接各轮，供记录）

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


def set_tool_generation(model, tokenizer, *, greedy: bool, max_new_tokens: int, temperature: float = 1.0):
    """每一轮生成的配置。停在 ``` 之后，好把代码块交出去执行。"""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    end = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    common = dict(max_new_tokens=max_new_tokens, eos_token_id=[im_end, end],
                  pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.0)
    if greedy:
        return GenerationConfig(do_sample=False, **common)
    return GenerationConfig(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, **common)


@torch.no_grad()
def rollout_with_tools(model, tokenizer, questions, *, device, gen_config, max_turns=3, max_tool_output=600,
                       tool_timeout=5.0):
    """对一批题逐条跑多轮工具轨迹。实现清楚优先，按题串行（0.5B 上够快）。返回 [Trajectory]。"""
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    trajectories = []
    for question in questions:
        messages = build_messages(question)
        base_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        traj = Trajectory(prompt_ids=list(base_ids))
        prev_ids = list(base_ids)
        for turn in range(max_turns):
            input_ids = torch.tensor([prev_ids], device=device)
            attn = torch.ones_like(input_ids)
            out = model.generate(input_ids=input_ids, attention_mask=attn, generation_config=gen_config)
            new_ids = out[0, len(prev_ids):].tolist()
            if new_ids and new_ids[-1] in (im_end, tokenizer.convert_tokens_to_ids("<|endoftext|>")):
                gen_ids = new_ids[:-1]          # 去掉结尾的 <|im_end|>，由下面统一补
                hit_eos = True
            else:
                gen_ids, hit_eos = new_ids, False
            traj.token_ids.extend(gen_ids)
            traj.mask.extend([1] * len(gen_ids))
            assistant_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            traj.text += assistant_text
            traj.turns += 1

            code = CODE_RE.search(assistant_text)
            has_answer = "####" in assistant_text
            # 结束条件：给了答案、或自然 EOS 且没有要执行的代码、或已是最后一轮
            if has_answer or (hit_eos and not code) or turn == max_turns - 1:
                traj.token_ids.append(im_end)
                traj.mask.append(1)            # 结尾的 <|im_end|> 算模型生成
                traj.finished = has_answer or hit_eos
                break
            # 执行代码，把结果作为一条 user 消息拼回去（工具返回，mask=0）
            result, is_error = run_python(code.group(1), timeout=tool_timeout)
            traj.tool_calls += 1
            traj.tool_errors += int(is_error)
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": f"```output\n{result}\n```"})
            full = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            # full 相对上一次渲染新增的部分 = <|im_end|>（结束 assistant）+ 整个 user 工具轮 + 新的 assistant 提示头
            tool_ids = full[len(prev_ids) + len(gen_ids):]
            traj.token_ids.extend(tool_ids)
            traj.mask.extend([0] * len(tool_ids))   # 工具返回和模板 token 都不算 loss
            traj.text += f"\n```output\n{result}\n```\n"
            prev_ids = full
        trajectories.append(traj)
    return trajectories


def collate_trajectories(trajectories, pad_id: int) -> dict:
    """多条轨迹拼成一个前向 batch：prompt 左填充、response 右填充，response_mask 用各自的 mask
    （工具返回的 token 在这里就是 0，所以 loss 和 logprob 都不会算到它们）。与 rollout.collate 结构一致。"""
    lp = max(t.prompt_len for t in trajectories)
    lr = max(len(t.token_ids) for t in trajectories)
    b = len(trajectories)
    input_ids = torch.full((b, lp + lr), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((b, lp + lr), dtype=torch.long)
    response_mask = torch.zeros((b, lr), dtype=torch.float32)
    for i, t in enumerate(trajectories):
        p, r = t.prompt_ids, t.token_ids
        input_ids[i, lp - len(p):lp] = torch.tensor(p)
        input_ids[i, lp:lp + len(r)] = torch.tensor(r)
        attention_mask[i, lp - len(p):lp + len(r)] = 1
        response_mask[i, :len(r)] = torch.tensor(t.mask, dtype=torch.float32)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
            "responses": input_ids[:, lp:].clone(), "response_mask": response_mask, "prompt_len": lp}
