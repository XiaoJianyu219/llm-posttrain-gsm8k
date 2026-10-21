"""把几次评测放进一张表。第一项当基线，其余每项和它做 McNemar 配对检验。

  python summarize_evals.py base=<eval 目录> sft=<eval 目录> dpo=<eval 目录> grpo=<eval 目录>

每个目录里要有 eval_gsm8k.py 写出的 summary.json 和 samples.jsonl（同一批题）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from compare_evals import load, mcnemar_exact


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    items = [arg.split("=", 1) for arg in sys.argv[1:]]
    if len(items) < 2 or any(len(it) != 2 for it in items):
        raise SystemExit(__doc__)
    summaries = {name: json.loads((Path(d) / "summary.json").read_text(encoding="utf-8")) for name, d in items}
    samples = {name: load(str(Path(d) / "samples.jsonl")) for name, d in items}
    base_name = items[0][0]
    keys = sorted(set.intersection(*(set(s) for s in samples.values())))
    print(f"题数 {len(keys)}（所有评测共同的题）；p 值是和 {base_name} 的 McNemar 精确检验\n")
    print(f"{'方法':<8}{'严格':>9}{'宽松':>9}{'带格式':>9}{'截断':>8}{'平均长度':>10}{'严格 p':>10}{'宽松 p':>10}")
    for name, _ in items:
        s = summaries[name]
        cells = []
        for metric in ("strict", "flexible"):
            if name == base_name:
                cells.append("-")
                continue
            a = [samples[base_name][k][f"correct_{metric}"] for k in keys]
            b = [samples[name][k][f"correct_{metric}"] for k in keys]
            only_new = sum((not x) and y for x, y in zip(a, b))
            only_base = sum(x and (not y) for x, y in zip(a, b))
            cells.append(f"{mcnemar_exact(only_new, only_base):.2g}")
        print(f"{name:<8}{s['acc_strict']:>9.4f}{s['acc_flexible']:>9.4f}{s['format_rate']:>9.4f}"
              f"{s['clip_ratio']:>8.3f}{s['response_tokens_mean']:>10.0f}{cells[0]:>10}{cells[1]:>10}")


if __name__ == "__main__":
    main()
