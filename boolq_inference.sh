python -u inference.py \
--random_seed=3407  \
--dataset="boolq" \
--model="gpt-3.5-turbo" \
--trainset_path="./dataset/GSM8K/train.jsonl" \
--demo_path="./distilled_demos_new/363-1509_boolq.txt" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--qes_limit=0 \
--zhipukey="baa531579b5f7a0783ea21f8dab22d41.28F42oFA2y9Q8o3A"