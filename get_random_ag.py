from datasets import load_dataset
from count_tokens import num_tokens_from_string
import os
import pdb
import random

os.environ['TIKTOKEN_CACHE_DIR'] = ''

demo = ""
model = "gpt-3.5-turbo"

dataset = load_dataset("ag_news")
print(len(dataset['train']))
indices = [random.randint(0, len(dataset['train'])) for _ in range(8)]
# pdb.set_trace()
for i in indices:
    demo += ("User: Classify the following text into one of these categories: World (0), Sports (1), Business (2), Sci/Tech (3). " + \
        dataset['train'][i]['text'] + "\nResponse: " + str(dataset['train'][i]['label']) + '\n\n')
        
# for data in dataset['train']:
#     # pdb.set_trace()
#     demo += ("User: " + "Passage: " + data['passage'] + " Question: " + data['question'] + \
#         "\nResponse: " + str(data['answer']) + '\n\n')  
#     num = num_tokens_from_string(demo, model)
#     print(f"num tokens: {num}")
#     if num >= 750:
#         break

num = num_tokens_from_string(demo, model)
print(f"num tokens: {num}")
print(demo)    