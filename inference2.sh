export CUDA_VISIBLE_DEVICES=0,1

python inference.py \
--random_seed=37 \
--dataset="gsm8k" \
--model="claude" \
--trainset_path="./dataset/GSM8K/train.jsonl" \
--demo_path="./distilled_demos_new/381-gsm8k_Llama-2-7b-chat-hf_4_2_trainsplit_24.txt" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--qes_limit=0 