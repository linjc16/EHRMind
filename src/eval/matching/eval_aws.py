import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
import pdb
import json

tqdm.pandas()

sys.path.append('./')

from src.utils.claude_aws import chat_haiku, chat_sonnet



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='claude-sonnet')
    parser.add_argument('--save_dir', type=str, default='results/matching/output')
    parser.add_argument("--data_path", type=str, default="data/local_index_search/matching/test.parquet")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    
    df = pd.read_parquet(args.data_path)

    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['label'].tolist()
    data_sources = df['data_source'].tolist()
    
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
            if args.model_name == 'claude-sonnet':
                decoded = chat_sonnet(prompt)
            elif args.model_name == 'claude-haiku':
                decoded = chat_haiku(prompt)
        except Exception as e:
            print(e)
            decoded = ""
        
        output_dict[idx] = {
            'output': decoded,
            'label': targets[idx],
            'data_source': data_sources[idx]
        }
        
        if i % 100 == 0:
            with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
                json.dump(output_dict, f, indent=4)

        i += 1
    
    with open(os.path.join(args.save_dir, f'{args.model_name}.json'), 'w') as f:
        json.dump(output_dict, f, indent=4)