export CUDA_VISIBLE_DEVICES=0,1
export TIKTOKEN_CACHE_DIR=""

python -u distill2.py \
--random_seed=42 \
--dataset="gsm8k" \
--model="gpt-3.5-turbo" \
--trainset_path="./dataset/GSM8K/train.jsonl" \
--demo_path="./initial_demos/2070-2_gsm8k.txt" \
--save_path="./distilled_demos/" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--num_pairs=8