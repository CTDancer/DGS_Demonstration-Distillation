export CUDA_VISIBLE_DEVICES=0,1

# python -u inference_noscore.py \
# --random_seed=3407  \
# --dataset="boolq" \
# --model="gpt-3.5-turbo" \
# --trainset_path="./dataset/GSM8K/train.jsonl" \
# --demo_path="./initial_demos/1509_boolq.txt" \
# --max_tokens=4096 --api_time_interval=2 --temperature=0 \
# --multipath=1 \
# --qes_limit=0 \
# --zhipukey="baa531579b5f7a0783ea21f8dab22d41.28F42oFA2y9Q8o3A"

# python -u inference.py \
# --random_seed=3407  \
# --dataset="gsm8k" \
# --model="gpt-3.5-turbo" \
# --trainset_path="./dataset/GSM8K/train.jsonl" \
# --demo_path="./initial_demos/609_gsm8k.txt" \
# --max_tokens=4096 --api_time_interval=2 --temperature=0 \
# --multipath=1 \
# --qes_limit=0

python -u inference_noscore.py \
--random_seed=3407  \
--dataset="multiple_rc" \
--model="gpt-3.5-turbo" \
--trainset_path="./dataset/MultiRC/train.jsonl" \
--demo_path="./distilled_demos_new/2shot_rc.txt" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--qes_limit=0