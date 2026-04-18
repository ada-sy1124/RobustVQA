# data_process（退款诉求多模态数据管线）

当前主脚本：`get_data_spa.py`

`get_data_sqa.py` 仍可运行，但只是兼容入口，会转调到 `get_data_spa.py`。

## 1) 适配的数据逻辑

### 输入样本结构
每条样本至少包含三列：
- `<image>`：商品图像（可为 HF image dict / bytes / 本地图片路径）
- `<review>`：用户退款诉求文本
- `<score>`：评分（0~5，支持浮点）

### 混合数据构建规则（可选）
脚本支持从原始 Amazon 数据构建混合集：
- 真实低星带图评价：30%（默认 `score<=2`）
- 高星样本伪造愤怒退款诉求：70%（默认 `score>=4`）
- 打乱后输出

默认模式下，脚本认为你的混合数据集已经构建完成，直接读取 `<image,review,score>` 并生成训练/评测数据。

### Ground Truth 标注规则（固定）
按 `score` 直接映射：
- `score in [4,5]` -> `D`（拒绝退款）
- `score = 3` -> `C`（部分补偿）
- `score in [1,2]` -> `B`（退货退款）
- `score = 0` -> `A`（仅退款）

## 2) 输出字段

评测 JSONL 中包含：
- `prompt`
- `non_thinking_prompt`
- `question`
- `review`
- `score`
- `ground_truth`
- `choices`
- `label_desc`
- `sample_type`
- `is_adversarial`

训练表格中保留兼容字段：
- `images`
- `prompt`（`<image>...`）
- `ground_truth`
- `choices`
- `score`

## 3) 运行命令

### A. 你的“已构建数据集”直接转训练/评测格式（默认推荐）
```bash
python3 data_process/get_data_spa.py \
  --input data/your_mixed_dataset.jsonl \
  --train-table data/ecom_refund_score_rule_train.parquet \
  --test-table data/ecom_refund_score_rule_test.parquet \
  --train-jsonl data/ecom_refund_score_rule_train.jsonl \
  --test-jsonl data/ecom_refund_score_rule_test.jsonl
```

### B. 从原始 Amazon 数据按 30/70 构建混合数据并输出
```bash
python3 data_process/get_data_spa.py \
  --input data/amazon_raw_with_image.jsonl \
  --build-mixed \
  --synthesis-mode openai \
  --synthesis-model gpt-4.1-mini \
  --real-ratio 0.3 \
  --fake-ratio 0.7 \
  --low-score-max 2 \
  --high-score-min 4 \
  --train-table data/ecom_refund_score_rule_train.parquet \
  --test-table data/ecom_refund_score_rule_test.parquet \
  --train-jsonl data/ecom_refund_score_rule_train.jsonl \
  --test-jsonl data/ecom_refund_score_rule_test.jsonl
```

## 4) 查看样本

```bash
python3 data_process/查看数据.py --file data/ecom_refund_score_rule_test.jsonl --num 3
```

## 5) 依赖说明

- 读写 `.parquet` 需要 `pyarrow` 或 `fastparquet`
- 若本地无 parquet 依赖，可把输出后缀改为 `.jsonl` / `.csv` / `.tsv`
- 使用 `--synthesis-mode openai` 时，需要安装 `openai` 并设置 `OPENAI_API_KEY`
