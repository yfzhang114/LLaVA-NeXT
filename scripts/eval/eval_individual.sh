# conda activate eval

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# name=${1:-'/home/yueke/model/llavanext-openai_clip-vit-large-patch14-336-Qwen_Qwen2-0.5B-Instruct-mlp2x_gelu-llavaonevision-lr1e-4'}
# # nohup  bash scripts/eval_mme.sh >> eval_mme_qwen.log 2>&1 &
# # mme,mmerealworld,mmerealworld_cn,gqa_lite,vqav2_val_lite,realworldqa,ocrbench,textvqa_val_lite,websrc_val,chartqa_lite,ai2d_lite,docvqa_val_lite,pope_adv
# for checkpoint in "$name"/checkpoint-* 
# do

# accelerate launch --num_processes 8 --main_process_port 12380 -m lmms_eval \
#     --model  llava    --model_args pretrained=$name,conv_template=qwen_1_5,model_name=llava_qwen\
#     --tasks   pope_adv \
#     --batch_size 1     \
#     --output_path ./logs/ 
# done

# nohup bash scripts/eval/eval_individual.sh >> eval_mme_qwen_05B_mmpreference.log 2>&1 &
conda activate eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

name=${1:-'/dev/shm/model/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-mmpreference'}
base=$(basename "$name")
# 指定最多 n 个 checkpoint 同时运行
n=1  # 你可以根据需要更改此值
base_port=12389  # 基础端口号

checkpoint_count=0  # 计数器


output_path=./logs/$base/$(basename "$name")

if [ -d "$output_path" ]; then
    echo "Output path already exists, skipping: $output_path"
    continue  # 跳过当前循环
fi



echo "Evaluating Model: $name"
port_count=1
while true; do
    port=$((base_port + port_count))
    if ! lsof -i -P -n | grep ":$port" > /dev/null; then
        break  # Exit the loop if the port is free
    fi
    port_count=$((port_count + 1))  # Increment to check the next port
done

# 启动评估
accelerate launch --num_processes 8 --main_process_port $port -m lmms_eval \
    --model llava --model_args pretrained=$name,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks mme,mmerealworld,mmerealworld_cn,gqa_lite,vqav2_val_lite,realworldqa,ocrbench,textvqa_val_lite,websrc_val,chartqa_lite,ai2d_lite,docvqa_val_lite,pope_adv \
    --batch_size 1 \
    --output_path $output_path &


wait  # 等待所有后台进程完成


# accelerate launch --num_processes 8 --main_process_port 12380 -m lmms_eval \
#     --model llava --model_args pretrained=/home/yueke/model/test,conv_template=v1,model_name=llava_openllm,conv_template=v1,model_name=llava_openllm \
#     --tasks mme \
#     --batch_size 1 \
#     --output_path ./logs 

# accelerate launch --num_processes=8 \
# -m lmms_eval \
# --model llava_onevision \
# --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen \
# --tasks mmerealworld \
# --batch_size 1 \
# --output_path ./logs/