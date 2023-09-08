export CUDA_VISIBLE_DEVICES=0,1

python inference.py \
--random_seed=3407  \
--dataset="gsm8k" \
--model="Baichuan2-13B-Chat" \
--trainset_path="./dataset/GSM8K/train.jsonl" \
--demo_path="./distilled_demos_new/557_gsm8k_16shot.txt" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--qes_limit=0