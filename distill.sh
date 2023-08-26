export CUDA_VISIBLE_DEVICES=0,1

# python distill.py \
# --random_seed=42 \
# --dataset="gsm8k" \
# --model="gpt-3.5-turbo" \
# --trainset_path="./dataset/GSM8K/train.jsonl" \
# --demo_path="./logdifference_results/gsm8k_Llama-2-7b-chat-hf_4_2_trainsplit_24.txt" \
# --save_path="./distilled_demos/" \
# --max_tokens=1024 --api_time_interval=2 --temperature=0 \
# --multipath=1 \
# --json_demo

python distill.py \
--random_seed=37 \
--dataset="squad" \
--model="gpt-3.5-turbo" \
--trainset_path=" " \
--demo_path="./initial_demos/squad_6shot.txt" \
--save_path="./distilled_demos/" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1

# python distill.py \
# --random_seed=42 \
# --dataset="MATH" \
# --model="gpt-3.5-turbo" \
# --trainset_path="/shared/dqwang/scratch/tongchen/MATH" \
# --demo_path="./distilled_demos/math_1.txt" \
# --max_tokens=1024 --api_time_interval=2 --temperature=0.7 \
# --multipath=1