export CUDA_VISIBLE_DEVICES=0,1

# python -u autocompressor_inference.py \
# --random_seed=37  \
# --dataset="boolq" \
# --demo_path="./distilled_demos_new/156-645_boolq.txt" \
# --qes_limit=0

python -u autocompressor_inference.py \
--random_seed=37  \
--dataset="multiple_rc" \
--demo_path="./initial_demos/718_rc.txt" \
--qes_limit=0