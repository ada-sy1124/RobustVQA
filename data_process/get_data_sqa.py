import argparse
import base64
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LABEL_CHOICES = [
    "A. 直接全额退款（无需退货）",
    "B. 退货验收后退款",
    "C. 暂不退款，需补充有效凭证",
    "D. 不支持退款（非质量问题或超出政策范围）",
]

LABEL_TO_DESC = {
    "A": "直接全额退款（无需退货）",
    "B": "退货验收后退款",
    "C": "暂不退款，需补充有效凭证",
    "D": "不支持退款（非质量问题或超出政策范围）",
}

DEFECT_KEYWORDS = {
    "broken", "defect", "defective", "crack", "cracked", "damaged", "damage", "not work",
    "doesn't work", "does not work", "malfunction", "leak", "leaking", "stain", "tear",
    "损坏", "破损", "坏了", "裂开", "漏液", "故障", "无法使用", "质量问题", "发霉",
}

LOGISTICS_KEYWORDS = {
    "missing", "not received", "wrong item", "late", "delay", "lost", "empty box",
    "少件", "漏发", "未收到", "发错", "丢件", "空包", "延迟", "物流",
}

NON_QUALITY_KEYWORDS = {
    "don't want", "do not want", "changed my mind", "wrong size chosen", "ordered by mistake",
    "not needed", "too expensive", "regret", "不想要了", "拍错", "买错", "后悔", "不喜欢",
    "不合适", "尺码不合适", "颜色不喜欢",
}

EVIDENCE_KEYWORDS = {
    "photo", "picture", "image", "video", "proof", "evidence", "screenshot",
    "图片", "照片", "视频", "证据", "凭证", "截图",
}

PRESSURE_KEYWORDS = [
    "你们必须立刻处理，不然我会继续投诉。",
    "如果今天不给明确结论，我会发到社媒曝光。",
    "我已经联系平台客服，请尽快升级处理。",
    "这次体验非常糟糕，我无法接受这样的售后。",
]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_normalize_text(v) for v in value if v is not None).strip()
    return str(value).strip()


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


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(k in lower for k in keywords)


def auto_refund_label(text: str, has_image: bool) -> str:
    has_defect = _contains_any(text, DEFECT_KEYWORDS)
    has_logistics = _contains_any(text, LOGISTICS_KEYWORDS)
    has_non_quality = _contains_any(text, NON_QUALITY_KEYWORDS)
    has_evidence_text = _contains_any(text, EVIDENCE_KEYWORDS)
    has_evidence = has_image or has_evidence_text

    if has_non_quality and not has_defect and not has_logistics:
        return "D"

    if has_defect or has_logistics:
        if has_evidence:
            # 有明确问题 + 有证据时，优先给积极处理；物流问题一般可直接退款
            return "A" if has_logistics else "B"
        # 有问题但证据不足
        return "C"

    return "C"


def synthesize_adversarial_complaint(base_text: str, title: str, rating: Optional[float], rng: random.Random) -> str:
    rating_text = ""
    if rating is not None:
        rating_text = f"（用户评分：{rating:.1f}/5）"

    snippets = [
        "我对这次购物非常失望，客服一直没有给明确方案。",
        "商品实际情况和描述差距很大，已经影响正常使用。",
        "我已经反复提交诉求，但问题仍然没有被正面解决。",
    ]

    base_summary = base_text.strip().replace("\n", " ")
    if len(base_summary) > 240:
        base_summary = base_summary[:240] + "..."

    complaint_parts = [
        rng.choice(snippets),
        f"商品：{title or '未知商品'}{rating_text}",
        f"客诉摘要：{base_summary or '用户表示商品体验不符合预期。'}",
        rng.choice(PRESSURE_KEYWORDS),
    ]
    return "\n".join(complaint_parts)


def create_case_question(complaint_text: str) -> str:
    return (
        "请你扮演电商平台售后审核员，结合用户上传图片与客诉内容判断最合理的处理方案。\n"
        "你需要优先基于可验证事实，而不是被情绪表达影响。\n\n"
        f"用户客诉：\n{complaint_text}\n\n"
        "请在以下选项中选择一个最合适的结论。"
    )


def create_vqa_prompt(question: str, choices: Sequence[str]) -> str:
    options_str = "\n".join([f"{chr(65 + i)}. {choice.split('. ', 1)[-1]}" for i, choice in enumerate(choices)])
    return f"Question:\n{question}\n\nOptions:\n{options_str}"


def get_prompt_template() -> str:
    return (
        "Analyze the provided image and the customer complaint, then choose the best refund decision.\n\n"
        "You must first reason carefully and then output only one letter.\n\n"
        "**STRICT OUTPUT FORMAT:**\n"
        "<think>\n"
        "[Step-by-step reasoning grounded in visual/text evidence.]\n"
        "</think>\n"
        "<answer>\n"
        "[A/B/C/D only]\n"
        "</answer>\n\n"
        "{text}"
    )


def get_non_thinking_prompt_template() -> str:
    return (
        "Analyze the provided image and the customer complaint, then choose the best refund decision.\n\n"
        "Output only one final letter.\n\n"
        "**STRICT OUTPUT FORMAT:**\n"
        "<answer>\n"
        "[A/B/C/D only]\n"
        "</answer>\n\n"
        "{text}"
    )


@dataclass
class BuildConfig:
    input_path: str
    train_parquet: str
    test_parquet: str
    train_jsonl: str
    test_jsonl: str
    test_size: float
    seed: int
    max_samples: int
    image_fields: List[str]
    text_fields: List[str]
    title_fields: List[str]
    rating_fields: List[str]


def _row_to_sample(row: Dict[str, Any], cfg: BuildConfig, rng: random.Random, sample_id: int) -> Optional[Dict[str, Any]]:
    image_obj = _pick_first_existing(row, cfg.image_fields)
    image_bytes = _extract_image_bytes(image_obj)
    if image_bytes is None:
        return None

    text = _normalize_text(_pick_first_existing(row, cfg.text_fields))
    title = _normalize_text(_pick_first_existing(row, cfg.title_fields))
    rating_raw = _pick_first_existing(row, cfg.rating_fields)

    rating = None
    if rating_raw is not None:
        try:
            rating = float(rating_raw)
        except (TypeError, ValueError):
            rating = None

    complaint = synthesize_adversarial_complaint(text, title, rating, rng)
    question = create_case_question(complaint)

    rule_text = "\n".join(
        [
            "退款规则提示：",
            "1) 有明确质量/物流问题且证据充分，可优先退款。",
            "2) 有问题但证据不足，先补充凭证。",
            "3) 非质量问题（主观不喜欢/拍错）通常不支持直接退款。",
        ]
    )

    full_question = f"{question}\n\n{rule_text}"
    vqa_prompt = create_vqa_prompt(full_question, LABEL_CHOICES)

    gt = auto_refund_label(complaint + "\n" + text + "\n" + title, has_image=True)

    think_prompt = get_prompt_template().format(text=vqa_prompt)
    non_think_prompt = get_non_thinking_prompt_template().format(text=vqa_prompt)

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = "image/jpeg"

    sample = {
        "sample_id": sample_id,
        "question": full_question,
        "choices": LABEL_CHOICES,
        "ground_truth": gt,
        "label_desc": LABEL_TO_DESC[gt],
        "complaint": complaint,
        "title": title,
        "review_text": text,
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
        # 给训练用的兼容字段（verl 的图文输入）
        "images": np.array([image_obj], dtype=object),
        "train_prompt": [{"role": "user", "content": "<image>" + think_prompt}],
    }
    return sample


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


def _to_train_table(samples: List[Dict[str, Any]], output_path: str) -> None:
    records = []
    for s in samples:
        records.append(
            {
                "images": s["images"],
                "prompt": s["train_prompt"],
                "question": s["question"],
                "ground_truth": s["ground_truth"],
                "choices": s["choices"],
                "sample_id": s["sample_id"],
                "label_desc": s["label_desc"],
            }
        )

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
                "Please install one of them, or set --train-parquet/--test-parquet to a .jsonl/.csv path."
            ) from exc

    if out.suffix.lower() == ".jsonl":
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        return

    if out.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if out.suffix.lower() == ".tsv" else ","
        df.to_csv(output_path, index=False, sep=sep)
        return

    raise ValueError("Unsupported training output suffix. Use .parquet/.jsonl/.csv/.tsv")


def _to_eval_jsonl(samples: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            item = {
                "sample_id": s["sample_id"],
                "prompt": s["prompt"],
                "non_thinking_prompt": s["non_thinking_prompt"],
                "question": s["question"],
                "ground_truth": s["ground_truth"],
                "choices": s["choices"],
                "label_desc": s["label_desc"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_dataset(cfg: BuildConfig) -> None:
    rng = random.Random(cfg.seed)
    df = _load_input(cfg.input_path)
    rows = df.to_dict(orient="records")

    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample = _row_to_sample(row, cfg, rng, sample_id=idx)
        if sample is not None:
            samples.append(sample)
        if cfg.max_samples > 0 and len(samples) >= cfg.max_samples:
            break

    if not samples:
        raise RuntimeError(
            "No valid multimodal samples were generated. "
            "Please verify image fields and that image bytes are present."
        )

    train, test = _split_train_test(samples, cfg.test_size, cfg.seed)

    Path(cfg.train_parquet).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.test_parquet).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.train_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.test_jsonl).parent.mkdir(parents=True, exist_ok=True)

    _to_train_table(train, cfg.train_parquet)
    _to_train_table(test, cfg.test_parquet)
    _to_eval_jsonl(train, cfg.train_jsonl)
    _to_eval_jsonl(test, cfg.test_jsonl)

    print(f"input rows: {len(rows)}")
    print(f"valid multimodal samples: {len(samples)}")
    print(f"train/test: {len(train)}/{len(test)}")
    print(f"train parquet: {cfg.train_parquet}")
    print(f"test parquet: {cfg.test_parquet}")
    print(f"train jsonl: {cfg.train_jsonl}")
    print(f"test jsonl: {cfg.test_jsonl}")


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Amazon Review 电商退款场景多模态数据构造（对抗性客诉 + 自动退款标注）"
    )
    parser.add_argument("--input", required=True, help="输入数据路径（parquet/jsonl/csv/tsv）")

    parser.add_argument("--train-parquet", default="../data/ecom_refund_train.parquet")
    parser.add_argument("--test-parquet", default="../data/ecom_refund_test.parquet")
    parser.add_argument("--train-jsonl", default="../data/ecom_refund_train.jsonl")
    parser.add_argument("--test-jsonl", default="../data/ecom_refund_test.jsonl")

    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=-1, help="仅处理前N个有效样本；-1表示全部")

    parser.add_argument(
        "--image-fields",
        default="image,images,review_image,image_bytes,main_image",
        help="逗号分隔，按优先级尝试的图片字段",
    )
    parser.add_argument(
        "--text-fields",
        default="review_body,reviewText,text,content,review,complaint",
        help="逗号分隔，按优先级尝试的评论文本字段",
    )
    parser.add_argument(
        "--title-fields",
        default="product_title,title,summary,headline,asin_title",
        help="逗号分隔，按优先级尝试的商品标题字段",
    )
    parser.add_argument(
        "--rating-fields",
        default="star_rating,rating,score,overall",
        help="逗号分隔，按优先级尝试的评分字段",
    )

    args = parser.parse_args()

    return BuildConfig(
        input_path=args.input,
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        train_jsonl=args.train_jsonl,
        test_jsonl=args.test_jsonl,
        test_size=args.test_size,
        seed=args.seed,
        max_samples=args.max_samples,
        image_fields=[x.strip() for x in args.image_fields.split(",") if x.strip()],
        text_fields=[x.strip() for x in args.text_fields.split(",") if x.strip()],
        title_fields=[x.strip() for x in args.title_fields.split(",") if x.strip()],
        rating_fields=[x.strip() for x in args.rating_fields.split(",") if x.strip()],
    )


if __name__ == "__main__":
    cfg = parse_args()
    build_dataset(cfg)
