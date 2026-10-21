"""比较同一批题上的两次评测（例如训练前 vs 训练后）。

  python compare_evals.py <基线 samples.jsonl> <新模型 samples.jsonl>

报告三件事：
1. 严格 / 宽松两种口径的准确率和差值；
2. McNemar 精确检验：同一批题，只看“一个对、一个错”的题，判断差值是否超出随机波动；
3. 严格口径下的错误拆成“没写 ####（格式错）”和“写了但数不对（算错）”，看提升主要来自哪一块。
"""

from __future__ import annotations

import json
import math
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["split"], r["index"]): r for r in rows}


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar 检验：b、c 是两种不一致的题数。"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    base, new = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(base) & set(new))
    if len(keys) != len(base) or len(keys) != len(new):
        print(f"注意：两份结果的题目不完全相同，只比较共同的 {len(keys)} 题")
    n = len(keys)
    print(f"题数 {n}\n基线：{sys.argv[1]}\n新的：{sys.argv[2]}\n")

    print("口径      基线      新的      差值      只有新的对  只有基线对  McNemar p")
    for metric in ("strict", "flexible"):
        a = [base[k][f"correct_{metric}"] for k in keys]
        b = [new[k][f"correct_{metric}"] for k in keys]
        only_new = sum((not x) and y for x, y in zip(a, b))
        only_base = sum(x and (not y) for x, y in zip(a, b))
        acc_a, acc_b = sum(a) / n, sum(b) / n
        print(f"{metric:<9} {acc_a:.4f}    {acc_b:.4f}    {acc_b - acc_a:+.4f}   {only_new:>8}    {only_base:>8}    "
              f"{mcnemar_exact(only_new, only_base):.2g}")

    print("\n严格口径下的错误拆分（题数）：")
    print("          格式错（没写 ####）  算错（写了但数不对）  宽松对但严格错")
    for name, d in (("基线", base), ("新的", new)):
        no_format = sum(d[k]["pred_strict"] is None for k in keys)
        wrong = sum(d[k]["pred_strict"] is not None and not d[k]["correct_strict"] for k in keys)
        format_only = sum(d[k]["correct_flexible"] and not d[k]["correct_strict"] for k in keys)
        print(f"{name:<8}  {no_format:>12}          {wrong:>12}          {format_only:>8}")


if __name__ == "__main__":
    main()
