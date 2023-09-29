python -u inference.py \
--random_seed=42  \
--dataset="gsm8k" \
--model="gpt-3.5-turbo" \
--trainset_path="" \
--demo_path="" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--qes_limit=0
