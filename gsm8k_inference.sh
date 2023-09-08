export CUDA_VISIBLE_DEVICES=0,1

python -u inference.py \
--random_seed=3407  \
--dataset="gsm8k" \
--model="gpt-3.5-turbo" \
--trainset_path="./dataset/GSM8K/train.jsonl" \
--demo_path="./initial_demos/half_gsm8k.txt" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--qes_limit=0