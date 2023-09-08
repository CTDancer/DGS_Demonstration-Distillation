from transformers import AutoTokenizer
from auto_compressor import AutoCompressorModel
import argparse
import json
import re
import time
import utils
import pdb

def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')
    
    # Load a model pre-trained on 6k tokens in 4 compression steps
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-2.7b-6k")
    model = AutoCompressorModel.from_pretrained("princeton-nlp/AutoCompressor-2.7b-6k").eval()

    # pdb.set_trace()
    try:
        with open(args.demo_path, "r") as file:
            demo = file.read()
    except FileNotFoundError:
        print("Your demo path doesn't exist. Please try another path.")
    # demo = '''1. Ant-Man and the Wasp - upcoming American superhero film based on Marvel Comics characters Scott Lang / Ant-Man and Hope van Dyne / Wasp. Is there going to be an Ant-Man 2 movie? Answer is True\n2. Kentucky - 70 mph speed limit on rural freeways as of 2007. Is there a speed limit on the Ohio River? Answer is True\n3. Marvel's Agents of S.H.I.E.L.D. - American television series created for ABC by Joss Whedon, Jed Whedon, and Maurissa Tancharoen, based on Marvel Comics organization S.H.I.E.L.D. Is Marvel Agents of Shield in the MCU? Answer is True\n4. '''

    demo_tokens = tokenizer(demo, return_tensors="pt").input_ids
    summary_vectors = model(demo_tokens, output_softprompt=True).softprompt
    print(f"Compressing {demo_tokens.size(1)} tokens to {summary_vectors.size(1)} summary vectors")
    
    dataloader = utils.create_dataloader(args)
    correct = 0
    wrong_list = []
    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)
    
    start = time.time()
    for count, qa in enumerate(dataloader):
        if args.qes_limit is not None and count == args.qes_limit:
            break
        message = ('\nFollow the given examples and answer the following question with true or false: ' + qa['question'] + ' Answer is: ')
        print(message)
        
        message_tokens = tokenizer(message, return_tensors="pt").input_ids
        # pdb.set_trace()
        output = model(message_tokens, softprompt=summary_vectors)
        if tokenizer.decode(output.logits[0,-1].argmax()) == '\n':
            answer = tokenizer.decode(output.logits[0,-2].argmax()).lower()
        else:
            answer = tokenizer.decode(output.logits[0,-1].argmax()).lower()
        print(f"answer is: {answer}")
        print(f"ground truth is: {qa['answer']}")
        if qa['answer'] == True:
            if 'yes' in answer or 'true' in answer:
                correct += 1
            else:
                wrong_list.append({'question': qa['question'], 'answer': answer, 'ground_truth': qa['answer']})
        else:
            if 'no' in answer or 'false' in answer:
                correct += 1
            else:
                wrong_list.append({'question': qa['question'], 'answer': answer, 'ground_truth': qa['answer']})
    
    end = time.time()
    print(f"Total correct number: {correct}")
    print(f"Correct Percentage: {correct / args.qes_limit}")
    print(f"Execution time: {end - start} seconds")
    
    summary_path = f"./summaries/one_prompt_round/{args.qes_limit}_{args.demo_path.split('/')[-1]}"
    with open(summary_path, "a") as f:
        f.write(f"Total correct number: {correct}\n")
        f.write(f"Correct Percentage: {correct / args.qes_limit}\n")
        f.write(f"Execution time: {end - start} seconds")
        
    wrong_list_path = f"./wrong_lists/{args.qes_limit}_{args.demo_path.split('/')[-1]}"
    with open(wrong_list_path, "a") as f:
        f.write(json.dumps(wrong_list, indent=4))
        
        
def arg_parser():
    parser = argparse.ArgumentParser(description="Inference with selected prompts.")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["boolq", "squad", "gsm8k", "svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith"], help="dataset to inference"
    )
    parser.add_argument(
        "--trainset_path", type=str, default="./dataset/GSM8K/train.jsonl", help="prompts to use"
    )
    parser.add_argument(
        "--demo_path", type=str, default="./distilled_demos/gsm8k_Llama-2-13b-chat-hf_4_2_trainsplit_42.txt", help="path to distilled demos"
    )
    parser.add_argument(
        "--QA_dir", type=str, default="./QA_records/", help="output directory for QA records"
    )
    parser.add_argument(
        "--wrong_list_dir", type=str, default="./wrong_lists/", help="output directory for wrong lists"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=512, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
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
    else:
        raise ValueError("dataset is not properly defined ...")

    return args


if __name__ == "__main__":
    main()
