export CUDA_VISIBLE_DEVICES=0,1

python -u summary_token.py \
--random_seed=42  \
--dataset="boolq" \
--demo_path="./initial_demos/645_boolq.txt" \
--qes_limit=0