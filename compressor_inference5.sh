export CUDA_VISIBLE_DEVICES=0,1

python -u autocompressor_inference.py \
--random_seed=37  \
--dataset="boolq" \
--demo_path="./distilled_demos_new/156-645_boolq.txt" \
--qes_limit=0