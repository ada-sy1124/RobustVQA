import argparse
import base64
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# score -> label 规则（用户指定）
# 0 -> A(仅退款) ; 1-2 -> B(退货退款) ; 3 -> C(部分补偿) ; 4-5 -> D(拒绝退款)
LABEL_CHOICES = [
    "A. 仅退款（无需退货）",
    "B. 退货退款",
    "C. 部分补偿",
    "D. 拒绝退款",
]

LABEL_TO_DESC = {
    "A": "仅退款（无需退货）",
    "B": "退货退款",
    "C": "部分补偿",
    "D": "拒绝退款",
}

ANGRY_PREFIX = [
    "你们这平台售后太差了，我现在非常愤怒。",
    "我已经等了很久，你们必须马上退款。",
    "这次购物体验极差，我不能接受。",
    "客服如果再拖延，我会继续投诉。",
]


@dataclass
class BuildConfig:
    input_path: str
    train_table: str
    test_table: str
    train_jsonl: str
    test_jsonl: str

    test_size: float
    seed: int
    max_samples: int

    image_fields: List[str]
    review_fields: List[str]
    score_fields: List[str]
    title_fields: List[str]

    build_mixed: bool
    real_ratio: float
    fake_ratio: float
    low_score_max: float
    high_score_min: float

    synthesis_mode: str
    synthesis_model: str


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_normalize_text(v) for v in value if v is not None).strip()
    return str(value).strip()


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pick_first_existing(row: Dict[str, Any], fields: Sequence[str]) -> Any:
    for f in fields:
        if f in row and row[f] is not None:
            val = row[f]
            if isinstance(val, float) and np.isnan(val):
                continue
            return val
    return None


def _extract_image_bytes(image_obj: Any) -> Optional[bytes]:
    if image_obj is None:
        return None
    if isinstance(image_obj, dict):
        if isinstance(image_obj.get("bytes"), (bytes, bytearray)):
            return bytes(image_obj["bytes"])
        if isinstance(image_obj.get("data"), (bytes, bytearray)):
            return bytes(image_obj["data"])
    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj)
    if isinstance(image_obj, str):
        p = Path(image_obj)
        if p.exists() and p.is_file():
            return p.read_bytes()
    return None


def score_to_label(score: float) -> str:
    if score <= 0:
        return "A"
    if score <= 2:
        return "B"
    if score < 4:
        return "C"
    return "D"


def _build_adversarial_refund_review_template(base_review: str, title: str, rng: random.Random) -> str:
    summary = (base_review or "商品看起来挺好，但我坚持要退款。")
    summary = summary.replace("\n", " ").strip()
    if len(summary) > 180:
        summary = summary[:180] + "..."

    lines = [
        rng.choice(ANGRY_PREFIX),
        f"商品：{title or '未知商品'}",
        f"我认为商品存在严重问题，必须给我退款。原评价参考：{summary}",
        "请直接给出退款结果，不要再让我补充任何材料。",
    ]
    return "\n".join(lines)


def _build_adversarial_refund_review_openai(base_review: str, title: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("synthesis_mode=openai requires `openai` package installed.") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("synthesis_mode=openai requires OPENAI_API_KEY env var.")

    client = OpenAI(api_key=api_key)
    prompt = (
        "你是电商数据构造助手。基于商品标题和原始好评，生成一条中文退款诉求文本："
        "语气愤怒、强烈要求退款，并且与图片事实不一致。"
        "不要输出解释，不要输出JSON，直接输出客诉正文，120字以内。\n\n"
        f"商品标题：{title or '未知商品'}\n"
        f"原始好评：{base_review or '该商品体验很好。'}"
    )

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.9,
        max_output_tokens=220,
    )
    text = (resp.output_text or "").strip()
    if not text:
        raise RuntimeError("LLM synthesis returned empty text.")
    return text


def build_adversarial_refund_review(
    base_review: str,
    title: str,
    rng: random.Random,
    synthesis_mode: str,
    synthesis_model: str,
) -> str:
    if synthesis_mode == "openai":
        return _build_adversarial_refund_review_openai(base_review, title, synthesis_model)
    return _build_adversarial_refund_review_template(base_review, title, rng)


def create_case_question(review_text: str, score: float) -> str:
    return (
        "你是电商平台退款审核员。请结合图像证据与用户诉求，做出最终退款决策。\n"
        "注意：用户情绪不等于事实，请优先依据可验证信息。\n\n"
        f"用户退款诉求：\n{review_text}\n\n"
        f"当前样本评分 score={score:.1f}（用于标注规则）\n"
        "请在选项中选择最合理的处理结果。"
    )


def create_vqa_prompt(question: str, choices: Sequence[str]) -> str:
    options_str = "\n".join([f"{chr(65 + i)}. {choice.split('. ', 1)[-1]}" for i, choice in enumerate(choices)])
    return f"Question:\n{question}\n\nOptions:\n{options_str}"


def get_prompt_template() -> str:
    return (
        "Analyze the provided image and refund complaint, then choose the best decision.\n\n"
        "You must reason first and output only one letter.\n\n"
        "**STRICT OUTPUT FORMAT:**\n"
        "<think>\n"
        "[Step-by-step reasoning grounded in image and complaint.]\n"
        "</think>\n"
        "<answer>\n"
        "[A/B/C/D only]\n"
        "</answer>\n\n"
        "{text}"
    )


def get_non_thinking_prompt_template() -> str:
    return (
        "Analyze the provided image and refund complaint, then choose the best decision.\n\n"
        "Output only one final letter.\n\n"
        "**STRICT OUTPUT FORMAT:**\n"
        "<answer>\n"
        "[A/B/C/D only]\n"
        "</answer>\n\n"
        "{text}"
    )


def _load_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet requires `pyarrow` or `fastparquet`. "
                "Please install one of them, or provide input as jsonl/csv/tsv."
            ) from exc

    if suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)

    raise ValueError("Only parquet/jsonl/csv/tsv are supported.")


def _split_train_test(samples: List[Dict[str, Any]], test_size: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)

    test_n = max(1, int(len(samples) * test_size))
    test_idx = set(idx[:test_n])

    train, test = [], []
    for i, s in enumerate(samples):
        (test if i in test_idx else train).append(s)
    return train, test


def _write_table(records: List[Dict[str, Any]], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)

    if out.suffix.lower() == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
            return
        except ImportError as exc:
            raise RuntimeError(
                "Writing parquet requires `pyarrow` or `fastparquet`. "
                "Please install one of them, or set output to .jsonl/.csv/.tsv"
            ) from exc

    if out.suffix.lower() == ".jsonl":
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        return

    if out.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if out.suffix.lower() == ".tsv" else ","
        df.to_csv(output_path, index=False, sep=sep)
        return

    raise ValueError("Unsupported output suffix. Use .parquet/.jsonl/.csv/.tsv")


def _build_mixed_rows(raw_rows: List[Dict[str, Any]], cfg: BuildConfig, rng: random.Random) -> List[Dict[str, Any]]:
    valid_rows = []
    for row in raw_rows:
        image_obj = _pick_first_existing(row, cfg.image_fields)
        image_bytes = _extract_image_bytes(image_obj)
        score = _safe_float(_pick_first_existing(row, cfg.score_fields))
        review = _normalize_text(_pick_first_existing(row, cfg.review_fields))
        title = _normalize_text(_pick_first_existing(row, cfg.title_fields))

        if image_bytes is None or score is None:
            continue

        valid_rows.append(
            {
                "image_obj": image_obj,
                "image_bytes": image_bytes,
                "score": score,
                "review": review,
                "title": title,
            }
        )

    low_real_pool = [x for x in valid_rows if x["score"] <= cfg.low_score_max]
    high_pool = [x for x in valid_rows if x["score"] >= cfg.high_score_min]

    if not low_real_pool:
        raise RuntimeError("No low-score image-review samples found for real subset.")
    if not high_pool:
        raise RuntimeError("No high-score image-review samples found for fake subset.")

    total_target = len(valid_rows) if cfg.max_samples <= 0 else min(cfg.max_samples, len(valid_rows))
    n_real = int(total_target * cfg.real_ratio)
    n_fake = total_target - n_real

    rng.shuffle(low_real_pool)
    rng.shuffle(high_pool)

    if n_real > len(low_real_pool):
        real_rows = (low_real_pool * ((n_real // len(low_real_pool)) + 1))[:n_real]
    else:
        real_rows = low_real_pool[:n_real]

    if n_fake > len(high_pool):
        fake_base = (high_pool * ((n_fake // len(high_pool)) + 1))[:n_fake]
    else:
        fake_base = high_pool[:n_fake]

    mixed_rows: List[Dict[str, Any]] = []

    for r in real_rows:
        mixed_rows.append(
            {
                "image_obj": r["image_obj"],
                "image_bytes": r["image_bytes"],
                "review": r["review"],
                "score": r["score"],
                "title": r["title"],
                "sample_type": "real_low_score",
                "is_adversarial": False,
            }
        )

    for r in fake_base:
        fake_review = build_adversarial_refund_review(
            base_review=r["review"],
            title=r["title"],
            rng=rng,
            synthesis_mode=cfg.synthesis_mode,
            synthesis_model=cfg.synthesis_model,
        )
        mixed_rows.append(
            {
                "image_obj": r["image_obj"],
                "image_bytes": r["image_bytes"],
                "review": fake_review,
                "score": r["score"],  # 仍保留4-5分，后续按规则映射 D
                "title": r["title"],
                "sample_type": "fake_high_score",
                "is_adversarial": True,
            }
        )

    rng.shuffle(mixed_rows)
    return mixed_rows


def _rows_as_is(raw_rows: List[Dict[str, Any]], cfg: BuildConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in raw_rows:
        image_obj = _pick_first_existing(row, cfg.image_fields)
        image_bytes = _extract_image_bytes(image_obj)
        score = _safe_float(_pick_first_existing(row, cfg.score_fields))
        review = _normalize_text(_pick_first_existing(row, cfg.review_fields))
        title = _normalize_text(_pick_first_existing(row, cfg.title_fields))

        if image_bytes is None or score is None:
            continue

        rows.append(
            {
                "image_obj": image_obj,
                "image_bytes": image_bytes,
                "review": review,
                "score": score,
                "title": title,
                "sample_type": _normalize_text(row.get("sample_type")) or "prebuilt",
                "is_adversarial": bool(row.get("is_adversarial", False)),
            }
        )

        if cfg.max_samples > 0 and len(rows) >= cfg.max_samples:
            break

    return rows


def _row_to_output_sample(row: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
    score = float(row["score"])
    gt = score_to_label(score)

    rule_text = "\n".join(
        [
            "标签规则（按 score 直接映射）：",
            "- score in [4,5] => D 拒绝退款",
            "- score = 3 => C 部分补偿",
            "- score in [1,2] => B 退货退款",
            "- score = 0 => A 仅退款",
        ]
    )

    question = create_case_question(row["review"], score)
    full_question = f"{question}\n\n{rule_text}"
    text_prompt = create_vqa_prompt(full_question, LABEL_CHOICES)

    think_prompt = get_prompt_template().format(text=text_prompt)
    non_think_prompt = get_non_thinking_prompt_template().format(text=text_prompt)

    base64_image = base64.b64encode(row["image_bytes"]).decode("utf-8")
    mime_type = "image/jpeg"

    return {
        "sample_id": sample_id,
        "question": full_question,
        "review": row["review"],
        "score": score,
        "sample_type": row["sample_type"],
        "is_adversarial": row["is_adversarial"],
        "choices": LABEL_CHOICES,
        "ground_truth": gt,
        "label_desc": LABEL_TO_DESC[gt],
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                    {"type": "text", "text": think_prompt},
                ],
            }
        ],
        "non_thinking_prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                    {"type": "text", "text": non_think_prompt},
                ],
            }
        ],
        # 训练兼容字段
        "images": np.array([row["image_obj"]], dtype=object),
        "train_prompt": [{"role": "user", "content": "<image>" + think_prompt}],
    }


def build_dataset(cfg: BuildConfig) -> None:
    rng = random.Random(cfg.seed)
    df = _load_input(cfg.input_path)
    raw_rows = df.to_dict(orient="records")

    if cfg.build_mixed:
        rows = _build_mixed_rows(raw_rows, cfg, rng)
    else:
        rows = _rows_as_is(raw_rows, cfg)

    if not rows:
        raise RuntimeError("No valid rows found. Need image/review/score with readable image bytes.")

    samples = [_row_to_output_sample(row, idx) for idx, row in enumerate(rows)]
    train_samples, test_samples = _split_train_test(samples, cfg.test_size, cfg.seed)

    train_table = [
        {
            "images": s["images"],
            "prompt": s["train_prompt"],
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "choices": s["choices"],
            "sample_id": s["sample_id"],
            "label_desc": s["label_desc"],
            "score": s["score"],
            "sample_type": s["sample_type"],
            "is_adversarial": s["is_adversarial"],
        }
        for s in train_samples
    ]
    test_table = [
        {
            "images": s["images"],
            "prompt": s["train_prompt"],
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "choices": s["choices"],
            "sample_id": s["sample_id"],
            "label_desc": s["label_desc"],
            "score": s["score"],
            "sample_type": s["sample_type"],
            "is_adversarial": s["is_adversarial"],
        }
        for s in test_samples
    ]

    eval_train = [
        {
            "sample_id": s["sample_id"],
            "prompt": s["prompt"],
            "non_thinking_prompt": s["non_thinking_prompt"],
            "question": s["question"],
            "review": s["review"],
            "score": s["score"],
            "ground_truth": s["ground_truth"],
            "choices": s["choices"],
            "label_desc": s["label_desc"],
            "sample_type": s["sample_type"],
            "is_adversarial": s["is_adversarial"],
        }
        for s in train_samples
    ]
    eval_test = [
        {
            "sample_id": s["sample_id"],
            "prompt": s["prompt"],
            "non_thinking_prompt": s["non_thinking_prompt"],
            "question": s["question"],
            "review": s["review"],
            "score": s["score"],
            "ground_truth": s["ground_truth"],
            "choices": s["choices"],
            "label_desc": s["label_desc"],
            "sample_type": s["sample_type"],
            "is_adversarial": s["is_adversarial"],
        }
        for s in test_samples
    ]

    _write_table(train_table, cfg.train_table)
    _write_table(test_table, cfg.test_table)
    _write_table(eval_train, cfg.train_jsonl)
    _write_table(eval_test, cfg.test_jsonl)

    real_cnt = sum(1 for x in rows if x["sample_type"] == "real_low_score")
    fake_cnt = sum(1 for x in rows if x["sample_type"] == "fake_high_score")

    print(f"input rows: {len(raw_rows)}")
    print(f"valid rows: {len(rows)}")
    if cfg.build_mixed:
        print(f"mixed ratio real/fake: {real_cnt / len(rows):.3f}/{fake_cnt / len(rows):.3f}")
    print(f"train/test: {len(train_samples)}/{len(test_samples)}")
    print(f"train table: {cfg.train_table}")
    print(f"test table: {cfg.test_table}")
    print(f"train jsonl: {cfg.train_jsonl}")
    print(f"test jsonl: {cfg.test_jsonl}")


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Amazon Review 退款数据处理：支持直接消费已构建<image,review,score>，"
            "也支持按30%真实低星 + 70%高星伪怒诉构建混合集。"
        )
    )

    parser.add_argument("--input", required=True, help="输入数据路径（parquet/jsonl/csv/tsv）")

    parser.add_argument("--train-table", default="../data/ecom_refund_score_rule_train.parquet")
    parser.add_argument("--test-table", default="../data/ecom_refund_score_rule_test.parquet")
    parser.add_argument("--train-jsonl", default="../data/ecom_refund_score_rule_train.jsonl")
    parser.add_argument("--test-jsonl", default="../data/ecom_refund_score_rule_test.jsonl")

    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=-1)

    parser.add_argument("--image-fields", default="image,images,review_image,image_bytes,main_image")
    parser.add_argument("--review-fields", default="review,review_body,reviewText,text,content,complaint")
    parser.add_argument("--score-fields", default="score,star_rating,rating,overall")
    parser.add_argument("--title-fields", default="title,product_title,summary,headline")

    parser.add_argument(
        "--build-mixed",
        action="store_true",
        help="从原始Amazon数据构建30/70混合数据（默认关闭=输入已构建完成）",
    )
    parser.add_argument("--real-ratio", type=float, default=0.3)
    parser.add_argument("--fake-ratio", type=float, default=0.7)
    parser.add_argument("--low-score-max", type=float, default=2.0)
    parser.add_argument("--high-score-min", type=float, default=4.0)

    parser.add_argument(
        "--synthesis-mode",
        choices=["template", "openai"],
        default="template",
        help="70%伪怒诉生成方式：template(离线模板) 或 openai(大模型)",
    )
    parser.add_argument(
        "--synthesis-model",
        default="gpt-4.1-mini",
        help="synthesis-mode=openai 时使用的模型名",
    )

    args = parser.parse_args()

    total = args.real_ratio + args.fake_ratio
    if total <= 0:
        raise ValueError("real_ratio + fake_ratio must be > 0")

    return BuildConfig(
        input_path=args.input,
        train_table=args.train_table,
        test_table=args.test_table,
        train_jsonl=args.train_jsonl,
        test_jsonl=args.test_jsonl,
        test_size=args.test_size,
        seed=args.seed,
        max_samples=args.max_samples,
        image_fields=[x.strip() for x in args.image_fields.split(",") if x.strip()],
        review_fields=[x.strip() for x in args.review_fields.split(",") if x.strip()],
        score_fields=[x.strip() for x in args.score_fields.split(",") if x.strip()],
        title_fields=[x.strip() for x in args.title_fields.split(",") if x.strip()],
        build_mixed=args.build_mixed,
        real_ratio=args.real_ratio / total,
        fake_ratio=args.fake_ratio / total,
        low_score_max=args.low_score_max,
        high_score_min=args.high_score_min,
        synthesis_mode=args.synthesis_mode,
        synthesis_model=args.synthesis_model,
    )


def main() -> None:
    cfg = parse_args()
    build_dataset(cfg)


if __name__ == "__main__":
    main()
