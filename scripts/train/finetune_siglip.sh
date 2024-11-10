export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

LLM_VERSION="/data/model/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
# nohup bash scripts/train/finetune_siglip.sh >> finetune_siglip_qwen2_7b_llava_sft.log 2>&1 &

# Set environment variables
NUM_GPUS=8  # Set the number of GPUs based on your hardware configuration
NNODES=1    # Set the number of nodes (usually 1 for single-node training)
RANK=0      # Set the rank of the current node (usually 0 for single node)
ADDR="127.0.0.1"  # Set the address of the master node (use 127.0.0.1 for local training)
PORT=12345   # Set the communication port (can be any unused port)

# Print the values of the environment variables to confirm they are set correctly
echo "NUM_GPUS: ${NUM_GPUS}"
echo "NNODES: ${NNODES}"
echo "RANK: ${RANK}"
echo "ADDR: ${ADDR}"
echo "PORT: ${PORT}"

PROMPT_VERSION="qwen_1_5"

MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-llava-sft"
echo "BASE_RUN_NAME: ${MID_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /data/llava_sft/llava_v1_5_mix665k.json \
    --image_folder /data/llava_sft/ \
    --pretrain_mm_mlp_adapter="/home/yueke/model/projectors/qwen_7b_mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/dev/shm/model/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8196 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn