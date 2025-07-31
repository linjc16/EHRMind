import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
import pdb
import json

tqdm.pandas()

sys.path.append('./')

from src.utils.gpt_azure import gpt_chat_4o, gpt_chat_35_msg, gpt_chat_o1_mini, gpt_chat_o3_mini



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    parser.add_argument('--save_dir', type=str, default='results/medcalc')
    parser.add_argument("--data_path", type=str, default="data/local_index_search/medcalc/test.parquet")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    df = pd.read_parquet(args.data_path)
    
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['Ground Truth Answer'].tolist()
    calids = df['Calculator ID'].tolist()
    upper_limits = df['Upper Limit'].tolist()
    lower_limit = df['Lower Limit'].tolist()
    categories = df['Category'].tolist()
    
    i = 0
    output_dict = {}
    for idx, prompt in enumerate(tqdm(inputs)):
        # extract the prompt from "<|im_start|>user" to <|im_start|>assistant
        prompt = prompt.split("<|start_header_id|>user<|end_header_id|>", 1)[1]
        prompt = prompt.split("<|start_header_id|>assistant<|end_header_id|>", 1)[0]

        prompt = prompt.replace("<|eot_id|>", "")
        prompt = prompt.strip()
        
        prompt = prompt.replace("{", "(")
        prompt = prompt.replace("}", ")")

        try:
            if args.model_name == 'gpt-4o':
                decoded = gpt_chat_4o(prompt)
            elif args.model_name == 'gpt-35':
                decoded = gpt_chat_35_msg(prompt)
            elif args.model_name == 'o1-mini':
                decoded = gpt_chat_o1_mini(prompt)
            elif args.model_name == 'o3-mini':
                prompt = prompt.replace('str(short_and_direct_answer_of_the_question)', 'str(short_and_direct_answer_of_the_question, with no units, no explanation, and no surrounding text.)')
                decoded = gpt_chat_o3_mini(prompt)
        except Exception as e:
            print(e)
            decoded = ""
        
        output_dict[idx] = {
            "generated_text": decoded,
            "target": targets[idx],
            'calid': calids[idx],
            'upper_limit': upper_limits[idx],
            'lower_limit': lower_limit[idx],
            'category': categories[idx]
        }
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(output_dict, f, indent=4)

        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)