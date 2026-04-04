set -x

step=194

local_dir="/home/RobustVQA-RL-ckpt/global_step_${step}/actor"  # 切换为对应的文件夹
hf_path="/home/RobustVQA-RL-ckpt/global_step_${step}/actor/huggingface"

output_path="/home/RobustVQA-RL-ckpt-step${step}-ckpt"

python3 legacy_model_merger.py merge \
    --backend=fsdp \
    --local_dir=$local_dir \
    --hf_model_path=$hf_path \
    --target_dir=$output_path

# merge lora
# python3 merge_lora.py \
#     --base_model=$output_lora_path \
#     --lora_path=$lora_path \
#     --merged_model_path=$output_path