import argparse
import json
from pathlib import Path

import pandas as pd


def inspect_parquet(path: Path, n: int) -> None:
    df = pd.read_parquet(path)
    print(f"=== Parquet: {path} ===")
    print(f"rows: {len(df)}, cols: {len(df.columns)}")
    print("columns:")
    for c in df.columns:
        print(f"- {c}")

    print(f"\n=== head({n}) ===")
    print(df.head(n))

    if len(df) > 0:
        row = df.iloc[0].to_dict()
        print("\n=== first sample preview ===")
        for k in ["sample_id", "score", "ground_truth", "label_desc", "sample_type", "is_adversarial", "question", "choices"]:
            if k in row:
                print(f"{k}: {row[k]}")


def inspect_jsonl(path: Path, n: int) -> None:
    print(f"=== JSONL: {path} ===")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break

    print(f"preview rows: {len(rows)}")
    for i, row in enumerate(rows):
        print(f"\n--- sample {i} ---")
        print("sample_id:", row.get("sample_id"))
        print("score:", row.get("score"))
        print("ground_truth:", row.get("ground_truth"))
        print("label_desc:", row.get("label_desc"))
        print("sample_type:", row.get("sample_type"))
        print("is_adversarial:", row.get("is_adversarial"))
        print("choices:", row.get("choices"))
        q = row.get("question", "")
        print("question:", (q[:220] + "...") if len(q) > 220 else q)


def main() -> None:
    parser = argparse.ArgumentParser(description="查看退款场景数据集样本")
    parser.add_argument("--file", required=True, help="待查看文件路径（.parquet / .jsonl）")
    parser.add_argument("--num", type=int, default=3, help="预览样本数")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    if path.suffix.lower() == ".parquet":
        inspect_parquet(path, args.num)
    elif path.suffix.lower() == ".jsonl":
        inspect_jsonl(path, args.num)
    else:
        raise ValueError("only .parquet or .jsonl is supported")


if __name__ == "__main__":
    main()
