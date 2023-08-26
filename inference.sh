export CUDA_VISIBLE_DEVICES=0,1

python inference.py \
--random_seed=3407 \
--dataset="squad" \
--model="gpt-3.5-turbo" \
--trainset_path=" " \
--demo_path="./distilled_demos/squad_306_6s.txt" \
--max_tokens=1024 --api_time_interval=2 --temperature=0 \
--multipath=1 \
--qes_limit=100 \
--multiple_prompting_rounds \