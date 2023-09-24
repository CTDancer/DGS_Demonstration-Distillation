import random
import sys
import json
import re
import time
import pdb
from datasets import load_dataset
import numpy as np
import torch
import openai
import requests
import tiktoken
import zhipuai

API_KEY = " "
# define for no solution if GPT cannot generate a valid solution
# here define a magic number for the convenience of variance calculation
NO_SOLUTION = '-10086'


# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# pass in a prompt and returns a response body contains response
def GPT3_request(model:str, input_prompt:list, max_tokens:int, time_interval, temperature=0.7, stop=None):
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = API_KEY
            resp = openai.Completion.create(
                model=model,
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
        # pause between each request to avoid rate limit
        time.sleep(time_interval)
    return resp

def GPT3_5_request(model:str, messages:list, max_tokens:int, time_interval=2, temperature=0.7, stop=None):
    ''''''
    API_KEY = "sk-AAMOOJ4kAVI8NeZKE066De9947874dF39aD8C804Dd89Be38"    # for api.dqwang.group
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = API_KEY
            openai.api_base = "https://api.dqwang.group/v1"
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {messages}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
        # pause between each request to avoid rate limit
        time.sleep(time_interval)
    
    # print("response is: ", resp)
    return resp['choices'][0]['message']['content']

def openai_ChatCompletion_create(**kwargs):
    url = 'https://api.dqwang.group/v1/chat/completions'
    headers = {
        "Authorization": "Bearer " + "sk-AAMOOJ4kAVI8NeZKE066De9947874dF39aD8C804Dd89Be38",
        "Content-type": "application/json",
        }
    data = json.dumps(kwargs)
    resp = requests.post(url, headers=headers, data=data)
    if 'choices' not in json.loads(resp.content):
        return 'NILL'
    return json.loads(resp.content)['choices'][0]['message']['content']

def claude(message):
    token = 'xoxp-5807096092167-5818713672245-5841552251043-21a356216b20a6ee028d42dafc890a91'
    
    def send_msg(token, message):
        sendurl = 'https://slack.com/api/chat.postMessage'
        data = {
            "token": token,
            "channel": "@Claude",
            "text": message
        }
        response = requests.post(sendurl, data=data)
        return response.text

    def receive_msg(token, timestamp):
        receiveurl = 'https://slack.com/api/conversations.history'
        data = {
            "token": token,
            "channel": "D05PZ7X729L",
            "oldest": timestamp
        }
        response = requests.post(receiveurl, data=data)
        return response.text

    msg = send_msg(token, message)
    data = json.loads(msg)
    timestamp = data['message']['ts']
    while True:
        time.sleep(10)
        response1 = json.loads(receive_msg(token, timestamp))['messages']
        if len(response1) != 0:
            response1 = response1[-1]['text']
        else:
            response1 = ''
        time.sleep(10)
        response2 = json.loads(receive_msg(token, timestamp))['messages']
        if len(response2) != 0:
            response2 = response2[-1]['text']
        else:
            response2 = ''
        if response2 != '' and response1 == response2:
            break
    return response1

def chatglm(message, args):
    zhipuai.api_key = args.zhipukey
    response = zhipuai.model_api.sse_invoke(
        model="chatglm_pro",
        prompt=message,
        temperature=args.temperature
    )
    # pdb.set_trace()
    response_data = [event.data for event in response.events()]
    response = ""
    for data in response_data:
        response += data
    # pdb.set_trace()
    return response

def load_data(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "gsm8k":
        dataset_path = "./dataset/GSM8K/train.jsonl" if args.distill is True else args.dataset_path
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    elif args.dataset == "squad":
        dataset_squad_v2 = load_dataset("squad_v2")
        for data in dataset_squad_v2['train']:
            questions.append(data['context'] + ' According to the given context, ' + data['question'])
            if len(data['answers']['text']) == 0:
                answers.append('unanswerable')
            else:
                answers.append(data['answers']['text'][0])
    elif args.dataset == "boolq":
        dataset_boolq = load_dataset("boolq")
        if args.distill is True:
            for data in dataset_boolq['train']:
                questions.append('Passage: ' + data['passage'] + ' Question: ' + data['question'] + '?')
                answers.append(data['answer'])
        else:
            for data in dataset_boolq['validation']:
                questions.append('Passage: ' + data['passage'] + ' Question: ' + data['question'] + '?')
                answers.append(data['answer'])
            # for data in dataset_boolq['validation']:
            #     questions.append(data['passage'] + ' ' + data['question'] + '?')
            #     answers.append(data['answer'])
    elif args.dataset == "ag_news":
        dataset_ag = load_dataset("ag_news")
        if args.distill is True:
            for data in dataset_ag['train']:
                questions.append("Classify the following text into one of these categories: World (0), Sports (1), Business (2), Sci/Tech (3). " + data['text'])
                answers.append(str(data['label']))
        else:
            for data in dataset_ag['test']:
                questions.append("Classify the following text into one of these categories: World (0), Sports (1), Business (2), Sci/Tech (3). " + data['text'])
                answers.append(str(data['label']))
    elif args.dataset == "multiple_rc":
        if args.distill is True:
            dataset_path = "./dataset/MultiRC/train.jsonl"
        else:
            dataset_path = "./dataset/MultiRC/val.jsonl"
        print(f"dataset_path: {dataset_path}")
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                # pdb.set_trace()
                question = f"Passage: \"{json_res['passage']['text']}\"\n"
                count = 0
                for question_data in json_res['passage']['questions']:
                    question += f"Question{count}: \"{question_data['question']}\""
                    
                    choices = []
                    # Loop through answer choices
                    for idx, answer in enumerate(question_data['answers']):
                        choices.append(chr(65 + idx) + ". " + answer['text'])
                    choices_str = ' '.join(choices)
                    question += f"Choices: {choices_str} \n"
                    count += 1
                
                answer = ""
                count = 0
                # pdb.set_trace()
                for question_data in json_res['passage']['questions']:
                    # correct_answers = [answer['label'] for idx, answer in enumerate(question_data['answers'])]
                    # answer += f"Question{count}: {str(correct_answers)} "
                    correct_answers = [chr(65 + idx) for idx, answer in enumerate(question_data['answers']) if answer['label'] == 1]
                    if len(correct_answers) == 0:
                        correct_answers = ['Z']
                    answer += f"Question{count}: {''.join(correct_answers)} "
                    count += 1
                questions.append(question)
                answers.append(answer)
                # passage = f"Passage: \"{json_res['passage']['text']}\"\n"
                # for question_data in json_res['passage']['questions']:
                #     ques = 'Question: ' + question_data['question']
                #     for answer in question_data['answers']:
                #         ques = 'Question: ' + question_data['question']
                #         ans = ''
                #         ques += '\nStudent\'s Answer: ' + answer['text'] + '. Is the student\'s answer correct or wrong?'
                #         question = passage + ques
                #         ans = 'Correct' if answer['label'] == 1 else 'Wrong'
                #         questions.append(question)
                #         answers.append(ans)
                    
    else:
        raise NotImplementedError

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers


def create_dataloader(args) -> list:
    '''Not a PyTorch dataloader. It supprts random index(slice) access'''
    set_random_seed(args.random_seed)
    questions, answers = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset


def create_input_prompt(args, qa_pairs, val_flag:bool)->str:
    '''return the combination of validation prompts and already-selected prompts'''
    x, y = [], []
    if val_flag:
        with open(args.prompt_path, encoding="utf-8") as f:
            json_data = json.load(f)
            json_data = json_data["prompt"]
            for line in json_data:
                x.append(line["question"])
                y.append(line["pred_ans"])
            if qa_pairs:
                for qa_pair in qa_pairs:
                    x.append(qa_pair["question"])
                    y.append(qa_pair["answer"])  
    else:
        if qa_pairs:
            for qa_pair in qa_pairs:
                x.append(qa_pair["question"])
                y.append(qa_pair["answer"])          

    index_list = list(range(len(x)))

    prompt_text = ""
    for i in index_list:
        prompt_text += x[i] + " " + y[i] + "\n\n"
    return prompt_text


def answer_extraction(args, responses):
    pred_ans = ""
    temp = responses
    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua", "csqa"):
        temp = re.findall(r'A|B|C|D|E', temp)
    elif args.dataset in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    elif args.dataset == 'squad':
        if "Response: " in temp:
            temp = temp[10:]
        temp = re.sub("\"|\'|\n", "", temp)
        temp = [temp]
    
    if len(temp) != 0:
        answer = temp[-1]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
        pred_ans = answer
    else:
        pred_ans = ""
    return pred_ans


def create_gpt_test_input_prompt(args) -> str:
    x, y, z = [], [], []
    with open(args.selected_prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        for line in json_data:
            z.append(line["dataset_idx"])

        with open(args.trainset_path, encoding="utf-8") as f2:
            for z_val in z:
                f2.seek(0)  # redirect fp to the beginning of file
                for i, line in enumerate(f2):
                    json_data = json.loads(line)
                    if i == z_val:
                        x.append(json_data["question"])
                        combine = json_data["answer"].split("\n")
                        one_prompt = ". ".join(combine)
                        one_prompt = one_prompt.replace('####', 'Therefore the answer is')
                        # one_prompt = one_prompt.replace('$', '')
                        y.append(one_prompt)

    index_list = list(range(len(x)))
    prompt_text = ""

    for i in index_list:
        prompt_text += "User: " + x[i] + "\n"+ "Response: "  + y[i] + "." + "\n\n"

    prompt_text = re.sub(r'<<.*?>>', '', prompt_text)   # delete calculator annotation

    return prompt_text

def get_qas(args):
    ''' get the question list and answer list from the given file '''
    if args.json_demo:
        qas = create_chat_completion_input_prompt(args, args.demo_path)
        questions = [qa["question"] for qa in qas]
        answers = [qa["answer"] for qa in qas]
    # elif args.multiple_lines:
    #     with open(args.demo_path, 'r') as file:
    #         qas = file.read()
    #     qas = qas.split('\n\n')
    #     questions = []
    #     answers = []
    #     for i in range(len(qas)):
    #         questions.append(qas[i].split('\n')[0][6:])
    #         answers.append(qas[i].split('Response: ')[1])
    # else:
    #     with open(args.demo_path, 'r') as file:
    #         qas = file.read()
    #     qas = qas.split('\n')
    #     questions = []
    #     answers = []
    #     for i in range(0, len(qas), 2):
    #         questions.append(qas[i][6:])
    #         answers.append(qas[i+1][10:])
    else:
        try:
            with open(args.demo_path, "r") as file:
                demo = file.read()
        except FileNotFoundError:
            print("Your demo path doesn't exist. Please try another path.")
    
    return questions, answers

def get_demos(args, questions=None, answers=None):
    ''' format the demonstration '''
    
    if args.json_demo:
        assert len(questions) == len(answers), "number of questions should be equal to number of answers"
        
        demonstrations = ""
        for i in range(len(questions)):
            demonstrations += ( "User: " + questions[i] + "\n" + "Response: " + answers[i] + "\n\n" )
    else:
        try:
            with open(args.demo_path, "r") as file:
                demonstrations = file.read()
        except FileNotFoundError:
            print("Your demo path doesn't exist. Please try another path.")
        
    return demonstrations

def get_initial_prompt(dataset):
    if dataset == 'gsm8k':
        initial_prompt = "Follow the given examples and answer the following question step by step.\
        Note that the last sentence in your response can ONLY start with `Therefore the answer is:`\n"
    elif dataset == 'squad':
        initial_prompt = "Follow the given examples and you only need to answer the following question step by step. \
        You should only use the information in the context to answer the question. \
        If you don't know the answer, just write 'unanswerable'. \
        Note that the last sentence in your response can ONLY start with `Therefore the answer is:`."
    else:
        initial_prompt = "Follow the given examples and answer the final question step by step.\
        Note that the last sentence in your response can ONLY start with `Therefore the answer is:`"
    
    return initial_prompt

def get_prompts():
    ''' define the prompts for distillation '''
    
    prompts = []
    # prompts.append("Rephrase or edit the demonstrations above so as to delete as much unimportant information as possible")
    # # prompts.append("Edit the given demonstrations to remove any redundant or repetitive information, ensuring the core message and logic remain intact if you can. ")
    # # prompts.append("Reduce wordiness in the demonstrations, making it more concise without losing its essential meaning and logic.")
    # prompts.append("Craft a succinct version of the demonstrations that omits redundant information while retaining its core essence and logic")
    # prompts.append("Revise the demonstrations using abbreviations and shortening where appropriate, ensuring the essential details and logic remain intact")
    # prompts.append("Can you please provide concise versions of the given demonstrations while retaining all the essential information, including key calculations and steps required to arrive at the answers? The goal is to make the solutions more streamlined without sacrificing clarity.")
    # # prompts.append("Edit each ANSWER so as to keep all calculation steps and delete their explanations if you can. " + \
    # #     "Then edit each QUESTION so as to omit as much redundant information as possible but must not lose any key information needed in the calculation steps of the answer if you can. ")
    # # prompts.append("Revise the demonstrations using abbreviations and shortening where appropriate, but must not lose any key information in each question needed in the calculation steps of the answer if you can. ")
    
    # for i in range(len(prompts)):
    #     prompts[i] = "These are the demonstrations for LLM chat completion. " + prompts[i] + \
        # "NOTE: " + \
        # "1. You must not remove the Q&A format. " + \
        # "2. You must ensure that all the final answers remain the same after edition."
    # prompts.append("First, edit or rephrase each Answer so as to omit any redundant information but must present all necessary computation steps. Then, edit or rephrase each question so as to keep all the information needed in the answer while omitting any redundant information. You must keep the Q&A format intact. If you think you can't do it, just output \"The demonstrations cannot be further distilled.\"")
    # prompts.append("These are some Question-Answer pairs. For each Answer, craft a succinct version of it that present all essential calculation steps while omitting any redundant information. You must remain each Question intact. You must ensure the final answer is still correct. If you think you can't do it, just output \"The demonstrations cannot be further distilled\"")
    # prompts.append("These are some Q-A pairs. For each content after 'Q: ', craft a succinct version of it that omits redundant information while retaining all the information and value needed in the corresponding 'A: '. You must not change the content after each 'A: '. There is no need for grammar correction. If you think you can't do it, just output \"The demonstrations cannot be further distilled\"")
    return prompts

def select_prompt(prompts, used_index, done):
    ''' select a prompt from prompts '''

    candidates = [x for x in list(range(len(prompts))) if x not in used_index]
    if len(candidates) == 0:
        done = True
        return None, done
    
    index = random.choice(candidates)    
    return prompts[index], done

def sample(args):
    ''' randomly sample a question-answer pair from the training set '''
    args.distill = True
    # pdb.set_trace()
    train_dataloader = create_dataloader(args)
    args.distill = False
    set_random_seed(args.random_seed)
    random.shuffle(train_dataloader)
    
    return train_dataloader[0]['question'], train_dataloader[0]['answer']


def create_completion_input_prompt(args) -> str:
    '''return formatted selected prompts for openai Completion'''
    x, y, z = [], [], []
    with open(args.selected_prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        for line in json_data:
            z.append(line["dataset_idx"])

        with open(args.trainset_path, encoding="utf-8") as f2:
            for z_val in z:
                f2.seek(0)  # redirect fp to the beginning of file
                for i, line in enumerate(f2):
                    json_data = json.loads(line)
                    if i == z_val:
                        x.append(json_data["question"])
                        combine = json_data["answer"].split("\n")
                        one_prompt = ". ".join(combine)
                        one_prompt = one_prompt.replace('####', 'Therefore the answer is')
                        # one_prompt = one_prompt.replace('$', '')
                        y.append(one_prompt)

    index_list = list(range(len(x)))
    prompt_text = ""

    for i in index_list:
        prompt_text += "Question: " + x[i] + "\n" + "Answer: " + y[i] + "." + "\n\n"

    prompt_text = re.sub(r'<<.*?>>', '', prompt_text)   # delete calculator annotation

    return prompt_text


def create_chat_completion_input_prompt(args, selected_prompt_path) -> list:
    '''return formatted selected prompts for openai ChatCompletion'''
    with open(selected_prompt_path, encoding="utf-8") as f1:
        selected_QAs = json.load(f1)    # selected_QAs is a python list consisting of dictionaries.

    decoder = json.JSONDecoder()
    with open(args.trainset_path, encoding="utf-8") as f2:
        trainset = f2.readlines()
        for qa_pair in selected_QAs:
            qa_pair_json = decoder.raw_decode(trainset[qa_pair["dataset_idx"]])[0]
            raw_answer = qa_pair_json["answer"]

            nextline_deleted = raw_answer.split('\n')
            merged_answer = ". ".join(nextline_deleted).replace('####', 'Therefore the answer is')
            processed_answer = re.sub(r'<<.*?>>', '', merged_answer)   # delete calculator annotation
            qa_pair["answer"] = processed_answer + "."

    return selected_QAs

def get_answers(trainset_path, question_list) -> list:
    decoder = json.JSONDecoder()
    count = len(question_list)
    answer_list = [None] * count
    with open(trainset_path, encoding="utf-8") as f:
        trainset = f.readlines()
        for qa in trainset:
            question = decoder.raw_decode(qa)[0]['question']
            if question in question_list:
                count -= 1
                answer_list[question_list.index(question)] = ". ".join(decoder.raw_decode(qa)[0]['answer'].split('\n')).replace('####', 'Therefore the answer is')
                if count == 0:
                    return answer_list
    return None

def create_chat_completion_input_prompt_from_APS(args) -> list:
    '''return formatted selected prompts from Active-Prompt Selections for openai ChatCompletion'''
    with open("./inference_prompts/gsm8k_k=10", encoding="utf-8") as f1:
        selected_QAs = json.load(f1)["prompt"]

    for qa_pair in selected_QAs:
        processed_answer = qa_pair["rationale"] + " Therefore the answer is " + qa_pair["pred_ans"] + "."
        processed_question = qa_pair["question"].removeprefix("User: ").removesuffix("\nResponse:")
        qa_pair["answer"] = processed_answer
        qa_pair["question"] = processed_question
    
    return selected_QAs

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")
  
def calculate_metrics(predictions, ground_truth):
    # Initialize variables to keep track of overall metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    exact_match_count = 0
    
    # Initialize variables for TP, FP, FN
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Iterate over each question's prediction and ground truth
    for prediction, truth in zip(predictions, ground_truth):
        # Initialize variables for this specific question
        tp = 0
        fp = 0
        fn = 0
        
        # Calculate TP, FP, FN for this question
        for char in prediction:
            if char in truth:
                tp += 1
            else:
                fp += 1
        for char in truth:
            if char not in prediction:
                fn += 1
        
        # Add TP, FP, FN for this question to the overall totals
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Check if the prediction matches the ground truth exactly
        exact_match = prediction == truth
        if exact_match:
            exact_match_count += 1
    
    # Calculate precision, recall, and F1 score using TP, FP, FN
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
    
    # Calculate the exact match rate
    num_questions = len(predictions)
    exact_match_rate = exact_match_count / num_questions
    
    return total_precision, total_recall, total_f1, exact_match_rate

                