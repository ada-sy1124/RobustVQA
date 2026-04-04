set -x

if [ $# -eq 0 ]; then
  echo "没有输入待评模型ID，请输入模型ID"
  exit 1
fi

model_id=$1

export MODEL_SERVER=127.0.0.1:8000
export MODEL_PATH=RobustVQA-RL-ckpt-step194-ckpt # 待测评模型
# export MODEL_PATH=Qwen2.5-VL-7B-Instruct

dataset="scienceqa_test_data.jsonl"  # 将之前生成的数据集复制一份到dataset文件夹下

python3 get_model_response.py \
--model_id $model_id \
--dataset $dataset

python3 print_metric.py \
--model_id $model_id \
--dataset $dataset