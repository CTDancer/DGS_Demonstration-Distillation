export CUDA_VISIBLE_DEVICES=0,1
export TIKTOKEN_CACHE_DIR=""

# python distill.py \
# --random_seed=42 \
# --dataset="ag_news" \
# --model="gpt-3.5-turbo" \
# --trainset_path="./dataset/GSM8K/train.jsonl" \
# --demo_path="./initial_demos/666_ag.txt" \
# --save_path="./distilled_demos_new/" \
# --max_tokens=4096 --api_time_interval=2 --temperature=0 \
# --multipath=1 \
# --num_pairs=8

# python -u distill2.py \
# --random_seed=42 \
# --dataset="boolq" \
# --model="gpt-3.5-turbo" \
# --trainset_path="./dataset/GSM8K/train.jsonl" \
# --demo_path="./initial_demos/3038-1_boolq.txt" \
# --save_path="./distilled_demos/" \
# --max_tokens=4096 --api_time_interval=2 --temperature=0 \
# --multipath=1 \
# --num_pairs=12 \
# --zhipukey="27c0bcb010c452b57ca1b72c7391df22.h1rY0WiHsCNjtzBY"

# python -u distill2.py \
# --random_seed=42 \
# --dataset="gsm8k" \
# --model="gpt-3.5-turbo" \
# --trainset_path="./dataset/GSM8K/train.jsonl" \
# --demo_path="./initial_demos/1061_gsm8k.txt" \
# --save_path="./distilled_demos/" \
# --max_tokens=4096 --api_time_interval=2 --temperature=0 \
# --multipath=1 \
# --num_pairs=8

python -u distill2.py \
--random_seed=42 \
--dataset="boolq" \
--model="gpt-3.5-turbo" \
--trainset_path="./dataset/MultiRC/train.jsonl" \
--demo_path="./initial_demos/1509_boolq.txt" \
--save_path="./distilled_demos_new/" \
--max_tokens=4096 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--num_pairs=8

# python distill.py \
# --random_seed=37 \
# --dataset="squad" \
# --model="gpt-3.5-turbo" \
# --trainset_path=" " \
# --demo_path="./initial_demos/squad_6shot.txt" \
# --save_path="./distilled_demos/" \
# --max_tokens=4096 --api_time_interval=2 --temperature=0 \
# --multipath=1

# python distill.py \
# --random_seed=42 \
# --dataset="MATH" \
# --model="gpt-3.5-turbo" \
# --trainset_path="/shared/dqwang/scratch/tongchen/MATH" \
# --demo_path="./initial_demos/8_algebra_math.txt" \
# --max_tokens=1024 --api_time_interval=2 --temperature=0.7 \
# --multipath=1 \
# --num_pairs=8