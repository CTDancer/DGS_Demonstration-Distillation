import argparse
import torch
import json
import time
import re
import os
import pdb
import utils
from count_tokens import num_tokens_from_string

def distill(previous_demos, question, answer=None, initial_prompt=None, args=None):
    if initial_prompt is None:
        if args.dataset == 'multiple_rc':
            initial_prompt = "Follow the given examples and answer each multiple-choice question. You should choose all the correct options."
        else:
            initial_prompt = "Follow the given examples and answer the final question step by step.\
                Note that the last sentence in your response can ONLY start with `Therefore the answer is:`"
    # print('*****************************')
    # print("question is: ", question)
    # print("answer is: ", answer)
    # print('*****************************')
    # print(f"previous demo: {previous_demos}\n")
    # print('*****************************')
    previous_length = num_tokens_from_string(previous_demos, 'gpt-3.5-turbo')
    previous_answer = answer
    
    first_time = True
    while True:
        if first_time is True: 
            distillation_prompt = f"""
I'm giving you several User-Response pairs, delimited by triple backticks.
```{previous_demos}```

# Task:
1. Distill the given User-Response pairs to be succinct while keeping the response logic and format and satisfying all the requirements.
2. After distillation, don't rush to give your result. Examine each User-Response pair and check whether each pair satisifies \
all the requirements. If not, you should modify your result accordingly.
3. Finally you can give your distillation result.

# Requirements:
1. In each User message, besides all the questions (and choices), preserve all the information related to these questions (and choices) and the Response message \
and then omit other unnecessary information. The information given in the User message should not be distilled into the Response message.
2. For each Response message, if there is a step-by-step derivation to the final answer in the initial version, you must preserve it INTACT in your distillation result. \
Otherwise, if there are no derivation steps in the initial version, you must NOT add derivation steps in your distillation result.
3. Must NOT change or omit the final answers in each Response message.
4. Must NOT omit questions (and choices) in each User message.
5. If the User-Response pair in the initial version has a step-by-step derivation to the final answer in the Response message, \
then you must also present this step-by-step derivation explicitly in the Response message in your distillation result.
6. The format of User messages and Response messages in your result must be the same as in the given version.

# Note:
If you think a User-Response pair does not need distillation, you should keep it intact instead of omitting it. \
Thus, the number of User-Response pairs in your distillation result should be the same as the given User-Response pairs.
"""      
        else:
            distillation_prompt = f"""
I'm giving you several User-Response pairs, delimited by triple backticks.
```{previous_demos}```

# Task:
1. Distill the given User-Response pairs to be succinct while keeping the response logic and format and satisfying all the requirements. \
If you think any distillation of the given User-Response pairs will fail to meet all the requirements, then just simply write 'No need for further distillation', \
so no distillation is needed and the task is over.
2. After distillation, don't rush to give your result. Examine each User-Response pair and check whether each pair satisifies \
all the requirements. If not, you should modify your result accordingly.
3. Finally you can give your distillation result.

# Requirements:
1. In each User message, besides all the questions (and choices), preserve all the information related to these questions (and choices) and the Response message \
and then omit other unnecessary information. The information given in the User message should not be distilled into the Response message.
2. For each Response message, if there is a step-by-step derivation to the final answer in the initial version, you must preserve it INTACT in your distillation result. \
Otherwise, if there are no derivation steps in the initial version, you must NOT add derivation steps in your distillation result.
3. Must NOT change or omit the final answers in each Response message.
4. Must NOT omit questions (and choices) in each User message.
5. If the User-Response pair in the initial version has a step-by-step derivation to the final answer in the Response message, \
then you must also present this step-by-step derivation explicitly in the Response message in your distillation result.
6. The format of User messages and Response messages in your result must be the same as in the given version.

# Note:
If you think a User-Response pair does not need distillation, you should keep it intact instead of omitting it. \
Thus, the number of User-Response pairs in your distillation result should be the same as the given User-Response pairs.
"""      
        # first_time = False
        messages_for_distillation = [
            # {"role": "system", "content": "You are a helpful assistant who will exactly follow user's orders."},
            {"role": "user", "content": (distillation_prompt)},
            # {"role": "system", "content": "You are a helpful assistant who will exactly follow user's orders."}
        ]
        stop = False
        while True:
            # print(f"first time is {first_time}")
            if args.model == "claude":
                distilled_demos = utils.claude(distillation_prompt)
            elif args.model == 'gpt-3.5-turbo':
                distilled_demos = utils.GPT3_5_request(
                    model=args.model, 
                    messages=messages_for_distillation,
                    max_tokens=args.max_tokens,
                    time_interval=args.api_time_interval,
                    temperature=args.temperature
                )
            elif args.model == "chatglm_pro":
                distilled_demos = utils.chatglm(messages_for_distillation, args)
            else:
                kwargs = {
                    "model": args.model,
                    "messages": messages_for_distillation,
                    "temperature": args.temperature
                }
                distilled_demos = utils.openai_ChatCompletion_create(**kwargs)
            print(f"distilled_demos is: {distilled_demos}\n")
            # distilled_length = len(distilled_demos.split())
            distilled_length = num_tokens_from_string(distilled_demos, 'gpt-3.5-turbo')
            # print(f"previous demo length: {previous_length}")
            print(f"distilled demo length: {distilled_length}\n")
            
            if distilled_length >= previous_length:
                if first_time is True:
                    messages_for_distillation.append({
                        "role": "assistant", "content": distilled_demos
                    })
                    messages_for_distillation.append({
                        "role": "user", "content": "Your distilled version is longer than the initial version. Please try again while meeting all the requirements."
                    })
                    print("Distillation Another Trial")
                    first_time = False
                    continue
                else:
                    stop = True
                    break

            if 'No need for further distillation' in distilled_demos:
                if distilled_length >= 50:
                    distilled_demos = distilled_demos.replace('No need for further distillation', '')
                # print("End Distillation")
                break

            first_time = False
            
            check_prompt = f"""
I'm giving you a text containing {args.num_pairs} User-Response pairs, delimited by three backticks.
For each User-Response pair, only those after 'Response: ' belong to the Response message and all the others belong to the User message which may contain 'Passage', 'Question', 'Student's answer' etc.
Your task is to score the text and tell me how many scores are deducted in total.
For variable N ranging from 1 to {args.num_pairs}, repeat the following process {args.num_pairs} times: 
Examine the Nth User-Response pair: (Only the messages after 'Response: ' belong to the Response message, and the others are belong to the User message.)
1. Whether the User message consists solely of questions without any additional content or context other than the questions? If it does, then deduct 10 points from the total score.
2. Whether the response message uses values or information that should be provided in the user message \
but are not explicitly provided in the user message to derive the final answer? \
If it does, then deduct 10 points from the total score.
3. Double check your score and make sure that your score is assigned according to these two criteria.

You don't need to care about the accuracy of the Response message.

Note that the last sentence in your response \
can ONLY start with `The score deducted in total is:`, and followed by the score deducted in total.

User-Response pairs: ```{distilled_demos}```
"""
#             check_prompt = f"""
# I'm giving you a text containing some User-Response pairs, delimited by three backticks.
# Your task is to score the text and tell me how many scores are deducted in total.
# You must follow the following scoring rules:
# 1. Examine each User-Response pair: \
# Whether the response message uses values or information that should be provided in the user message \
# but are not explicitly provided in the user message to derive the final answer? \
# If it does, then deduct 10 points from the total score.
# 2. Double check your scoring at the end to make sure that you have evaluated each pair appropriately.

# Note that the first sentence in your response \
# can ONLY start with `The score deducted in total is:`, and followed by the score deducted in total.

# User-Response pairs: ```{distilled_demos}```
# """

# 2. If it says "No need for further edition" but you think the initial version can be futher distilled while keeping the response logic and the format, \
# then deduct 50 points from the total score, and write "It can be further distilled". \
# Else if you think it do not need further distillation, then score=100.
# 3. If it is not in User-Response pair format and doesn't say "No need for further edition", \
# then deduct 50 points from the total score, and write "It is not in User-Response format".
# 4. For each User-Response pair in the distilled version, \
# if I only give you the information in its User message, not given extra knowledge, \
# can you get the same result as in its Response message? \
# If you can't, then deduct 10 points from the total score.
# 5. Otherwise, no points need to be deducted.

            messages_for_check = [
                # {"role": "system", "content": "You are a serious teacher."},
                {"role": "user", "content": (check_prompt)}
            ]
            if args.model == "claude":
                check_response = utils.claude(check_prompt)
            elif args.model == 'gpt-3.5-turbo':
                check_response = utils.GPT3_5_request(
                    model=args.model, 
                    messages=messages_for_check,
                    max_tokens=args.max_tokens,
                    time_interval=args.api_time_interval,
                    temperature=args.temperature
                )
            elif args.model == "chatglm_pro":
                check_response = utils.chatglm(messages_for_check, args)
            else:
                kwargs = {
                    "model": args.model,
                    "messages": messages_for_check,
                    "temperature": args.temperature
                }
                check_response = utils.openai_ChatCompletion_create(**kwargs)
            # print(f"check response is {check_response}")
            # check_score = int(re.findall(r'\d+', check_response.split("----------\n")[1])[0])
            # if check_score < 10:
            #     print("End Distilation")
            #     break
            # else:
            #     suggestion = check_response.split("----------")[-1]
            #     print(f"**********\nsuggestion is: {suggestion}")
            #     messages_for_distillation.append({
            #         "role": "assistant", "content": distilled_demos
            #     })
            #     messages_for_distillation.append({
            #         "role": "user", "content": "These are some comments on your distillation result: \n" + suggestion + "\nYou should distill the given User-Response pairs again based on the comments. \
            #         You must follow the requirements I required before. You need to present all the distilled User-Respone pairs, not just the pairs that need improvement. \
            #         If you think the initial version does not need further distillation, just write 'No need for further distillation' and no distillation is needed."
            #     })
            #     print("Distillation Another Trial")
            check_response = re.findall(r'\d+', check_response.split('\n')[-1])
            if len(check_response) == 0:
                check_score = 0
            else:
                check_score = int(check_response[0])
            # check_score = int(re.findall(r'\d+', check_response.split('\n')[0])[0])
            if check_score >= 10:
                if args.model == "claude":
                    utils.claude("Your previous distillation result has omitted necessary information or values in the User messages. Please try again. If you think the initial \
                    version does not need further distillation, just write 'No need for further distillation'.")
                else:
                    messages_for_distillation.append({
                        "role": "assistant", "content": distilled_demos
                    })
                    messages_for_distillation.append({
                        "role": "user", "content": "Your previous distillation result has omitted necessary information or values in the User messages. Please try again \
and make sure that your distillation result contains more necessary information and values in each User message this time than your previous result. \
Your distillation result this time must also meet all the requirements previously proposed. \
If you think the initial version does not need further distillation, just write 'No need for further distillation' and no distillation is needed."
                    })
                # print("Distillation Another Trial")
            else:
                # print("End Distillation")
                break
        
        if 'No need for further distillation' in distilled_demos or stop is True:
            break
        # 将出现次数最多的答案当成预测结果
        predictions = []
        messages_for_inference = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (initial_prompt + distilled_demos + '\nUser: ' + question)}
        ]
        # for i in range(0, args.multipath):
        #     prediction = utils.GPT3_5_request(
        #         model=args.model, 
        #         messages=messages_for_inference,
        #         max_tokens=args.max_tokens,
        #         time_interval=args.api_time_interval,
        #         temperature=args.temperature
        #     )
        #     print(f"prediction is: {prediction}\n")
        #     prediction = utils.answer_extraction(args, prediction).lstrip()
        #     print(f"Extracted answer: {prediction}\n")
        #     predictions.append(prediction)
        #     print("**************************")
        # prediction = max(predictions, key=predictions.count)
        prediction = utils.GPT3_5_request(
            model=args.model, 
            messages=messages_for_inference,
            max_tokens=args.max_tokens,
            time_interval=args.api_time_interval,
            temperature=args.temperature
        )
        # print(f"prediction is: {prediction}\n")
        # print("**************************")
        score_prompt = f"""
Question: ```{question}```

Student's answer:  ```{prediction}```

You should first read the given question and then read the student's answer. \
Take your time to organize and understand the logic of the student's answer. \
Your task is to provide a score out of 100 for the student's answer based on the following criteria:
1. Accuracy: whether the logic of the student's answer is correct and whether the final answer of the student's answer is correct
2. Relevance: how closely the student's answer aligns with the question's requirements
3. Coherence: whether the student's answer flow logically and make sense

You should also meet the following requirements:
- You should first explicitly analyze the question and the student's answer.
- Then, you should find all the mistakes in the student's answer if mistakes exist.
- If you've found mistakes in the student's answer, please give your solutions. \
After giving your solutions, check whether the student's answer is actually different from your solutions. \
If not, then your judgement may not be right, so review again.
- If the student's final answer is wrong or there is a critical mistake in the calculation that leads to an incorrect answer, the score should not be below 80. \
If there are no errors, the score should be close to 100. \
If there are minor errors which do not impact the correctness of the final answer, the score can be slightly lower but not below 90.
- You should assign a fair score based on whether the student's answer is actually correct or incorrect, \
rather than relying on appearances of correctness or incorrectness.

Note that the last sentence in your response can ONLY start with `Therefore the score is:` \
and followed by a score between 0 and 100.
"""
        score_message = [
            # {"role": "system", "content": "You are a serious teacher."},
            {"role": "user", "content": (score_prompt)}
        ]
        if args.model == "claude":
            response = utils.claude(score_prompt)
        elif args.model == 'gpt-3.5-turbo':
            response = utils.GPT3_5_request(
                model=args.model, 
                messages=score_message,
                max_tokens=args.max_tokens,
                time_interval=args.api_time_interval,
                temperature=args.temperature
            )
        elif args.model == "chatglm_pro":
            response = utils.chatglm(score_message, args)
        else:
            kwargs = {
                "model": args.model,
                "messages": score_message,
                "temperature": args.temperature
            }
            response = utils.openai_ChatCompletion_create(**kwargs)
        # print(f"SCORE RESPONSE: {response}")
        response_list = response.split('\n')
        score = 0
        for i in range(len(response_list)-1, 0, -1):
            if re.findall(r'\d+', response_list[i]):
                score = int(re.findall(r'\d+', response_list[i])[0])
                break
        # print(f"Score is {score}")
        if int(score) >= 90:
            # print("yes")
            previous_demos = distilled_demos
            previous_length = distilled_length
            previous_answer = prediction
        else:
            break
    
    if previous_answer is None:
        messages_for_inference = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (initial_prompt + previous_demos + '\nUser: ' + question)}
        ]
        if args.model == 'claude':
            previous_answer = utils.claude(messages_for_inference)
        elif args.model == 'gpt-3.5-turbo':
            previous_answer = utils.GPT3_5_request(
                model=args.model, 
                messages=messages_for_inference,
                max_tokens=args.max_tokens,
                time_interval=args.api_time_interval,
                temperature=args.temperature
            )
        elif args.model == "chatglm_pro":
            previous_answer = utils.chatglm(messages_for_inference, args)
        else:
            kwargs = {
                "model": args.model,
                "messages": messages_for_inference,
                "temperature": args.temperature
            }
            previous_answer = utils.openai_ChatCompletion_create(**kwargs)
        print(f"SCORE RESPONSE: {response}")
        print(f"Prediction is: {previous_answer}")
    
    return previous_demos, previous_length, previous_answer
            

def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')
    utils.set_random_seed(args.random_seed)
    used_index = []
    done = False
    select_prompt = True
    
    # dataloader = utils.create_dataloader(args)
    if args.json_demo:
        questions, answers = utils.get_qas(args)
        previous_demos = utils.get_demos(args, questions, answers)
    else:
        previous_demos = utils.get_demos(args)
    initial_prompt = utils.get_initial_prompt(args.dataset)
    question, answer = utils.sample(args)
    distilled_length_list = []
    for _ in range(50):
        previous_length = num_tokens_from_string(previous_demos, 'gpt-3.5-turbo')
        print(f"initial length: {previous_length}")
        distilled_demos, distilled_length, distilled_answer = distill(previous_demos, question, answer, initial_prompt, args)
        print(distilled_demos)
        print(f"Length: {distilled_length}")
        distilled_length_list.append(distilled_length)
    print(distilled_length_list)
    # dest = os.path.join(args.save_path, args.demo_path.split('/')[-1])
    # if os.path.exists(dest):
    #     with open(dest, "r") as file:
    #         demos = file.read()
    #     words_count = len(demos.split())
    
    # if os.path.exists(dest) and words_count < distilled_length:
    #     print("It is not shorter than the previous distilled version.")
    # else:
    #     with open(dest, "w") as file: 
    #         file.write(distilled_demos)

def arg_parser():
    parser = argparse.ArgumentParser(description="Inference with selected prompts.")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["multiple_rc", "ag_news", "boolq", "squad", "gsm8k","svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith"], help="dataset to inference"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./dataset/GSM8K/"
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
        "--multiple_lines", action='store_true', help='Use demonstrations that has multiple lines in Response message.'
    )
    parser.add_argument(
        "--num_pairs", type=int, default=8, help="number of User-Response pairs"
    )
    parser.add_argument(
        "--distill", type=bool, default=False, help="whether load training set"
    )
    parser.add_argument(
        "--zhipukey", type=str, default="", help='API key for zhipu'
    )

    args = parser.parse_args()

    if args.multipath > 1:
        args.temperature = 0.7
    else:
        args.temperature = 0
    print(f"Temperature: {args.temperature}")

    if args.dataset == "gsm8k":
        args.dataset_path = "./dataset/GSM8K/test.jsonl"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
    elif args.dataset == "asdiv":
        args.dataset_path = "./dataset/ASDiv/ASDiv.json"
    elif args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/dev_rand_split.jsonl"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/strategyQA/task.json"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_test.json"
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/MAWPS/AddSub.json"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/MAWPS/SingleEq.json"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MAWPS/MultiArith.json"
    elif args.dataset == 'squad':
        args.dataset_path = "squad_v2"
    elif args.dataset == 'boolq':
        args.dataset_path = "boolq"
    elif args.dataset == "ag_news":
        args.dataset_path = "ag_news"
    elif args.dataset == "multiple_rc":
        args.dataset_path == "./dataset/MultiRC/"
    else:
        raise ValueError("dataset is not properly defined ...")

    return args


if __name__ == "__main__":
    main()