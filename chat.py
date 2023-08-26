import argparse
import torch
import json
import time
import re
import os
import pdb
import sys
import utils
from distill import distill

def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')
    utils.set_random_seed(args.random_seed)

    demo = ""
    demo_list = []
    while True:
        # 获取用户输入
        print("First give some prompts, then ask the question.")
        messages = []
        for line in iter(input, ''):
            messages.append(line)
        question = messages[-1]
        prompts = ""
        for i in range(0, len(messages)-1):
            prompts += messages[i]
            prompts += '\n'
        user_message = prompts + question
        # print(f"User: {user_message}")

        # 当用户输入end，问答结束
        if user_message.lower() == "end":
            break

        # 获取回复
        if args.mes1 == True:
            # 第一种message是将前面所有问答的都当成prompt
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": demo + user_message}
            ]
        else:
            # 第二种是之前的问答仍然保持对话的形式
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            assert len(demo_list)%2 == 0, "The demo list has different number of User messages and Response."
            for i in range(0, len(demo_list), 2):
                messages.append({"role": "user", "content": demo_list[i][6:]})
                messages.append({"role": "assistant", "content": demo_list[i+1][10:]})
            messages.append({"role": "user", "content": user_message})
        # print(f"Message is: {messages}")
        # response = utils.GPT3_5_request(
        #     model=args.model, 
        #     messages=messages,
        #     max_tokens=args.max_tokens,
        #     time_interval=args.api_time_interval,
        #     temperature=args.temperature
        # )
        # print('*****************************')
        # print(f"Response is: {response}")

        # 将当前的User-Response蒸馏
        if prompts == "":
            demo += ('User: ' + question + '\n')
            demo += ('Response: ' + response + '\n')
            demo_list.append('User: ' + question + '\n')
            demo_list.append('Response: ' + response + '\n')
        else:
            # 每次只蒸馏当前的prompt
            start = time.time()
            distilled_prompts, distilled_length = distill(prompts, question, None, None, args)
            end = time.time()
            print(f"Distillation time: {end-start} seconds")
            print(f"Distilled prompts length = {distilled_length}")
            demo += ('User: ' + distilled_prompts + '\n' + question + '\n')
            demo += ('Response: ' + response + '\n')
            demo_list.append('User: ' + distilled_prompts + '\n' + question + '\n')
            demo_list.append('Response: ' + response + '\n')


def arg_parser():
    parser = argparse.ArgumentParser(description="Inference with selected prompts.")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["squad", "gsm8k","svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith"], help="dataset to inference"
    )
    parser.add_argument(
        "--trainset_path", type=str, default="./dataset/GSM8K/train.jsonl", help="prompts to use"
    )
    parser.add_argument(
        "--demo_path", type=str, default="./logdifference_results/gsm8k_baichuan7b_8-1_trainsplit-val.txt", help="path to demos"
    )
    parser.add_argument(
        "--save_path", type=str, default="./distilled_demos", help="path to save demos"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="model used for decoding."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./QA_records/", help="output directory for QA records"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=15, help="how many seconds to sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
    parser.add_argument(
        "--concat_length", type=int, default=4, help='Used for task last_letters, indicates length of last letter to concat, i.e. Elon Musk -> nk, use concat length of 2'
    )
    parser.add_argument(
        "--use_code_style_prompt", type=bool, default=False, help='Use code-style prompt as mentioned in paper for last_letters dataset'
    )
    parser.add_argument(
        "--json_demo", action='store_true', help='Use demonstrations or distilled demonstrations in json format'
    )
    parser.add_argument(
        "--mes1", action='store_true', help='Use the first kind of message'
    )
    parser.add_argument(
        "--way1", action='store_true', help='Use the first kind of distillation'
    )
        
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()