import json
import utils
import random
import re
from count_tokens import num_tokens_from_string

decoder = json.JSONDecoder()
demo = ""

with open("./dataset/GSM8K/train.jsonl", encoding="utf-8") as f2:
    trainset = f2.readlines()
    indices = [random.randint(0, len(trainset)) for i in range(2)]
    for index in indices:
        qa = decoder.raw_decode(trainset[index])[0]
        demo += "User: " + qa["question"] + '\n'
        raw_answer = qa["answer"]
        nextline_deleted = raw_answer.split('\n')
        merged_answer = ". ".join(nextline_deleted).replace('####', 'Therefore the answer is')
        processed_answer = re.sub(r'<<.*?>>', '', merged_answer)
        demo += "Response: " + processed_answer + '\n'

print(demo)
tokens = num_tokens_from_string(demo, "gpt-3.5-turbo")
print(tokens)