from transformers import AutoTokenizer
from auto_compressor import AutoCompressorModel
import pdb

# Load a model pre-trained on 6k tokens in 4 compression steps
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-2.7b-6k")
context = '''
1. Ant-Man and the Wasp - upcoming American superhero film based on Marvel Comics characters Scott Lang / Ant-Man and Hope van Dyne / Wasp. Is there going to be an Ant-Man 2 movie? Answer is True\n2. Kentucky - 70 mph speed limit on rural freeways as of 2007. Is there a speed limit on the Ohio River? Answer is True\n3. User: Marvel's Agents of S.H.I.E.L.D. - American television series created for ABC by Joss Whedon, Jed Whedon, and Maurissa Tancharoen, based on Marvel Comics organization S.H.I.E.L.D. Is Marvel Agents of Shield in the MCU? Answer is True\n
'''
context_tokens = tokenizer(context, return_tensors="pt").input_ids
# pdb.set_trace()
print(context_tokens.shape)