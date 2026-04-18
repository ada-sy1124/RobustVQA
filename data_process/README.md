# data_process（电商退换货多模态数据管线）

该目录已从 ScienceQA 处理逻辑改为 README 对应的电商退款场景：
- 基于 Amazon Review 风格字段构造样本。
- 自动合成对抗性客诉文本（情绪化 + 事实摘要）。
- 自动打退款决策标签（A/B/C/D）。
- 产出训练用表格文件 + 评测用 JSONL（含 `prompt` / `non_thinking_prompt`）。

## 1. 主脚本

`get_data_sqa.py`

虽然文件名保留历史命名，但内容已切换为电商退款数据构造器。

## 2. 输入支持

支持输入：`.parquet` / `.jsonl` / `.csv` / `.tsv`。

默认按优先级自动匹配字段：
- 图片字段：`image,images,review_image,image_bytes,main_image`
- 文本字段：`review_body,reviewText,text,content,review,complaint`
- 标题字段：`product_title,title,summary,headline,asin_title`
- 评分字段：`star_rating,rating,score,overall`

你可以通过参数覆盖这些字段名。

## 3. 标签定义

- `A`：直接全额退款（无需退货）
- `B`：退货验收后退款
- `C`：暂不退款，需补充有效凭证
- `D`：不支持退款（非质量问题或超出政策范围）

## 4. 运行示例

```bash
python3 data_process/get_data_sqa.py \
  --input data/amazon_reviews_with_image.jsonl \
  --train-parquet data/ecom_refund_train.parquet \
  --test-parquet data/ecom_refund_test.parquet \
  --train-jsonl data/ecom_refund_train.jsonl \
  --test-jsonl data/ecom_refund_test.jsonl
```

如果本地没有 `pyarrow/fastparquet`，将 `--train-parquet` / `--test-parquet` 改成 `.jsonl` 或 `.csv` 即可。

## 5. 查看样本

```bash
python3 data_process/查看数据.py --file data/ecom_refund_test.jsonl --num 3
```

