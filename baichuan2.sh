python -u inference_noscore.py \
--random_seed=37  \
--dataset="gsm8k" \
--model="Baichuan2-13B-Chat" \
--trainset_path="./dataset/GSM8K/train.jsonl" \
--demo_path="./distilled_demos_new/777-1233_gsm8k.txt" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--qes_limit=0