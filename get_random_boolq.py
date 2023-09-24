from datasets import load_dataset
from count_tokens import num_tokens_from_string
import os
import pdb
import random

os.environ['TIKTOKEN_CACHE_DIR'] = ''

demo = ""
model = "gpt-3.5-turbo"

dataset = load_dataset("boolq")
print(len(dataset['train']))
indices = [random.randint(0, len(dataset['train'])) for _ in range(2)]
for i in indices:
    demo += ("User: " + "Passage: " + dataset['train'][i]['passage'] + " Question: " + dataset['train'][i]['question'] + \
        "?\nResponse: " + str(dataset['train'][i]['answer']) + '\n\n') 
# for data in dataset['train']:
#     # pdb.set_trace()
#     demo += ("User: " + "Passage: " + data['passage'] + " Question: " + data['question'] + \
#         "\nResponse: " + str(data['answer']) + '\n\n')  
#     num = num_tokens_from_string(demo, model)
#     print(f"num tokens: {num}")
#     if num >= 750:
#         break

num = num_tokens_from_string(demo, model)
print(demo)    
print(f"num tokens: {num}")