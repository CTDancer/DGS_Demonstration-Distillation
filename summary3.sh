export CUDA_VISIBLE_DEVICES=2,3

# python -u summary_token.py \
# --random_seed=3407  \
# --dataset="boolq" \
# --demo_path="./initial_demos/645_boolq.txt" \
# --qes_limit=0

# python -u inference.py \
# --random_seed=3407  \
# --dataset="gsm8k" \
# --model="gpt-3.5-turbo" \
# --trainset_path="./dataset/GSM8K/train.jsonl" \
# --demo_path="./distilled_demos_new/812-1464_gsm8k.txt" \
# --max_tokens=4096 --api_time_interval=2 --temperature=0 \
# --multipath=1 \
# --qes_limit=0

python -u summary_token.py \
--random_seed=3407  \
--dataset="multiple_rc" \
--demo_path="./initial_demos/718_rc.txt" \
--qes_limit=0