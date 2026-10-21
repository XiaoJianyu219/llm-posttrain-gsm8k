"""下载 GSM8K，从训练集切出验证集，写成 verl 同款格式（jsonl + parquet）。

  python prepare_data.py --out-dir /root/autodl-tmp/p1/data/gsm8k                  # 默认从 GitHub 下载原始 jsonl
  python prepare_data.py --out-dir ... --source hf                                 # 走 HuggingFace（需要 datasets，可配 HF_ENDPOINT）
  python prepare_data.py --out-dir ... --source local --raw-dir <含 train.jsonl、test.jsonl 的目录>

验证集从训练集里按固定种子随机切 500 题：训练中的监控和选点只看它；
测试集 1319 题只在训练前、训练后各测一次（和检测项目里“阈值只在验证集上选”一个道理）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

from gsm8k import build_row, write_jsonl

GITHUB_RAW = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"


def load_raw(source: str, raw_dir: Path | None, save_dir: Path) -> dict:
    """返回 {"train": [...], "test": [...]}，每条是 {"question", "answer"}；原始文件另存一份到 save_dir。"""
    raw = {}
    if source == "hf":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")
        for split in ("train", "test"):
            raw[split] = [{"question": r["question"], "answer": r["answer"]} for r in ds[split]]
            write_jsonl(save_dir / f"{split}.jsonl", raw[split])
        return raw
    for split in ("train", "test"):
        if source == "local":
            text = (raw_dir / f"{split}.jsonl").read_text(encoding="utf-8")
        else:
            with urllib.request.urlopen(GITHUB_RAW.format(split=split), timeout=120) as resp:
                text = resp.read().decode("utf-8")
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")
        raw[split] = [json.loads(line) for line in text.splitlines() if line.strip()]
    return raw


def write_parquet(path: Path, rows: list) -> bool:
    """第 ② 步给 verl 用；没装 pandas / pyarrow 就跳过（训练和评测只读 jsonl）。"""
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(path, index=False)
        return True
    except ImportError:
        return False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", choices=["github", "hf", "local"], default="github")
    ap.add_argument("--raw-dir", type=Path, help="--source local 时，含 train.jsonl、test.jsonl 的目录")
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.source == "local" and args.raw_dir is None:
        raise SystemExit("--source local 需要 --raw-dir")

    out_dir = Path(args.out_dir)
    raw = load_raw(args.source, args.raw_dir, out_dir / "raw")
    order = list(range(len(raw["train"])))
    random.Random(args.seed).shuffle(order)
    val_idx = sorted(order[:args.val_size])
    train_idx = sorted(order[args.val_size:])

    splits = {
        "train": [build_row(raw["train"][i]["question"], raw["train"][i]["answer"], "train", i) for i in train_idx],
        "val": [build_row(raw["train"][i]["question"], raw["train"][i]["answer"], "val", i) for i in val_idx],
        "test": [build_row(r["question"], r["answer"], "test", i) for i, r in enumerate(raw["test"])],
    }
    parquet_ok = True
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        parquet_ok = write_parquet(out_dir / f"{name}.parquet", rows) and parquet_ok

    info = {
        "source": args.source, "seed": args.seed, "val_size": args.val_size,
        "counts": {k: len(v) for k, v in splits.items()},
        "raw_sha256": {s: sha256(out_dir / "raw" / f"{s}.jsonl") for s in ("train", "test")},
        "val_indices_in_raw_train": val_idx,
        "parquet_written": parquet_ok,
    }
    (out_dir / "split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train {info['counts']['train']} | val {info['counts']['val']} | test {info['counts']['test']} | "
          f"parquet {'yes' if parquet_ok else 'no (pip install pandas pyarrow)'} | -> {out_dir}")


if __name__ == "__main__":
    main()
