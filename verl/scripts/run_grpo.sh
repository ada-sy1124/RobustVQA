set -x

ulimit -n 65535

export CUDA_VISIBLE_DEVICES=0,1,2,3
export SWANLAB_LOG_DIR=swanlog
export SWANLAB_MODE=local
export HYDRA_FULL_ERROR=1
# export SWANLAB_API_KEY=xxx  # 如果SWANLAB_MODE为cloud, 则写上api_key

# verifier配置
export SELF_VERIFIER_SERVER=127.0.0.1:8000  # 切换一下
export SELF_VERIFIER_SERVER_NAME=Qwen2.5-VL-7B-Instruct

TRAIN_FILES=/home/output/RobustVQA/data/scienceqa_train_data.parquet
VAL_FILES=/home/output/RobustVQA/data/scienceqa_test_data.parquet

MODEL_PATH=/home/models/Qwen2.5-VL-7B-Instruct
OUTPUT_PATH=/home/output/RobustVQA-RL-ckpt  # ckpt保存路径

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=prompt \
    data.image_key=images \
    data.train_batch_size=64 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='right' \
    data.return_raw_chat=True \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=custom \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.default_local_dir=$OUTPUT_PATH \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.resume_mode="disable" \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    trainer.total_epochs=2 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 $@
