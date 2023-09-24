import json
import utils
import random
import re
from count_tokens import num_tokens_from_string

decoder = json.JSONDecoder()
demo = ""
# queslist = [
#     'Peggy is moving and is looking to get rid of her record collection',
#     'Hasan is packing up his apartment',
#     'Angelo and Melanie want to plan how many hours over the next week',
#     'A curry house sells curries that have varying levels of spice',
#     'Sam works at the Widget Factory, assembling Widgets',
#     'Ellie went to visit a circus with Sarah and they both got lost',
#     'Janet hires six employees. Four of them are warehouse workers',
#     'Alec is running for Class President'
# ]
queslist = [
    'Peggy is moving and is looking to get rid of her record collection',
    'A curry house sells curries that have varying levels of spice',
    'Ellie went to visit a circus with Sarah and they both got lost',
    'Janet hires six employees. Four of them are warehouse workers',
]
# queslist = [
#     'In 2004, there were 60 kids at a cookout',
#     'Zilla spent 7% of her monthly earnings on rent',
#     'If Buzz bought a pizza with 78 slices at a restaurant',
#     'Jame gets a raise to $20 per hour and works 40 hours a week',
#     'Mr. Gardner bakes 20 cookies, 25 cupcakes',
#     'A used car lot has 24 cars and motorcycles (in total) for sale',
#     'Norma takes her clothes to the laundry',
#     'Adam has an orchard. Every day for 30 days he picks 4 apples'
# ]
count = 0
with open("./dataset/GSM8K/train.jsonl", encoding="utf-8") as f2:
    trainset = f2.readlines()
    for qa in trainset:
        qa = decoder.raw_decode(qa)[0]
        for ques in queslist:
            if ques in qa['question']:                
                demo += "User: " + qa["question"] + '\n'
                raw_answer = qa["answer"]
                nextline_deleted = raw_answer.split('\n')
                merged_answer = ". ".join(nextline_deleted).replace('####', 'Therefore the answer is')
                processed_answer = re.sub(r'<<.*?>>', '', merged_answer)
                demo += "Response: " + processed_answer + '\n'
                count += 1
        if count == len(queslist):
            break

print(demo)
tokens = num_tokens_from_string(demo, "gpt-3.5-turbo")
print(tokens)