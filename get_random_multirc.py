import json
import utils
import random
import re
from count_tokens import num_tokens_from_string

decoder = json.JSONDecoder()
demo = ""
        
with open("./dataset/MultiRC/train.jsonl", encoding="utf-8") as f:
    trainset = f.readlines()
    indices = [random.randint(0, len(trainset)) for i in range(2)]
    for index in indices:
        json_res = decoder.raw_decode(trainset[index])[0]
        
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
        for question_data in json_res['passage']['questions']:
            correct_answers = [chr(65 + idx) for idx, answer in enumerate(question_data['answers']) if answer['label'] == 1]
            answer += f"Answer{count}: {''.join(correct_answers)} "
            count += 1
        
        demo += "User: " + question + "Response: " + answer + '\n\n'
        
        # passage = f"Passage: \"{json_res['passage']['text']}\"\n"
        # for question_data in json_res['passage']['questions']:
        #     for answer in question_data['answers']:
        #         ques = 'Question: ' + question_data['question']
        #         ans = ''
        #         ques += '\nStudents Answer: ' + answer['text']
        #         question = passage + ques
        #         ans = 'True' if answer['label'] == 1 else 'False'
        #         demo += "User: " + question + "\nJudgement is " + ans + '\n\n'

print(demo)
tokens = num_tokens_from_string(demo, "gpt-3.5-turbo")
print(tokens)