import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
from collections import defaultdict
import pdb

import sys
sys.path.append('./')
from src.dataset.medcalc.utils import get_balanced_validation_set



PROMPT = """You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score.
Here is the patient note:
```{note}```

Here is the task:
```{question}```"""

def make_prefix(dp):
    input_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n""" + PROMPT.format(note=dp['Patient Note'], question=dp['Question'])
    input_str += """\nPlease show your entire reasoning process in **a single** <think> </think> block (do not open or close the tag more than once). Your final response must be in JSON format within <answer> </answer> tags. For example,
<think>
[entire reasoning process here]
</think>
<answer>
{
    "answer": str(short_and_direct_answer_of_the_question),
} 
</answer>.

Do not output anything after the </answer> tag.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\nLet me solve this step by step.\n<think>"""

    return input_str






def load_medcalc_dataset():
    train_path = os.path.join(f'data/raw_data/medcalc/train_data.csv')
    full_train_data = load_dataset('csv', data_files=train_path, split='train')

    # Step 1: Get balanced validation set of 100 samples
    val_data, remaining_data = get_balanced_validation_set(full_train_data, 'Category', total_val_size=100)

    # Step 2: Remaining data becomes train_data, which we balance
    # train_data = balance_dataset_by_category(remaining_data, 'Category')
    train_data = remaining_data
    
    test_path = os.path.join(f'data/raw_data/medcalc/test_data.csv')
    test_data = load_dataset('csv', data_files=test_path, split='train')
    
    return train_data, val_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/local_index_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset', type=str, default='medcalc_sub')

    args = parser.parse_args()
    
    data_source = args.dataset
    
    train_data, val_data, test_data = load_medcalc_dataset()
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = make_prefix(example)
            solution = {
                "target": example['Ground Truth Answer'],
                "calid": example['Calculator ID'],
                "upper_limit": example['Upper Limit'],
                "lower_limit": example['Lower Limit'] 
            }
            if split == 'test':
                data_source_new = data_source + '_' + example['Category'] + '_' + split
            else:
                data_source_new = data_source + '_' + split
            
            data = {
                "data_source": data_source_new,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "medcalc",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'category': example['Category']
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val', data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', data_source), with_indices=True)
    
    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)

    val_dataset = concatenate_datasets([val_dataset, test_dataset])
    
    lengths_list = []
    for d in train_dataset:
        lengths_list.append(len(d['prompt'][0]['content'].split()))

    lengths_list_test = []
    for d in test_dataset:
        lengths_list_test.append(len(d['prompt'][0]['content'].split()))
    
    lengths_list_val = []
    for d in val_dataset:
        lengths_list_val.append(len(d['prompt'][0]['content'].split()))

    # remove the data larger than 1024
    max_tokens = 1024
    
    def filter_by_length(dataset, lengths):
        return dataset.select([i for i, l in enumerate(lengths) if l <= max_tokens])

    train_dataset = filter_by_length(train_dataset, lengths_list)
    
    print(f"Number of samples in train dataset: {len(train_dataset)}")
    print(f"Number of samples in test dataset: {len(test_dataset)}")
    print(f"Number of samples in val dataset: {len(val_dataset)}")
    
    # print(f"Average length of train dataset: {sum(lengths_list) / len(lengths_list)}")
    # print(f"Average length of test dataset: {sum(lengths_list_test) / len(lengths_list_test)}")
    # print(f"Average length of val dataset: {sum(lengths_list_val) / len(lengths_list_val)}")
    
    # print(f"Max length of train dataset: {max(lengths_list)}")
    # print(f"Max length of test dataset: {max(lengths_list_test)}")
    # print(f"Max length of val dataset: {max(lengths_list_val)}")
    
    
    local_dir = os.path.join(args.local_dir, args.dataset)
    hdfs_dir = os.path.join(args.hdfs_dir, args.dataset) if args.hdfs_dir is not None else None
    
    os.makedirs(local_dir, exist_ok=True)

    CATEGORY_MERGE_MAP = {
        "lab test": "lab",
        "lab": "lab",
        "dosage conversion": "dosage",
        "dosage": "dosage",
    }

    def save_by_category(dataset, split_name):
        category_to_examples = defaultdict(list)
        for example in dataset:
            raw_category = example['extra_info'].get('category', 'unknown')
            category = CATEGORY_MERGE_MAP.get(raw_category, raw_category)  # 映射归并
            category_to_examples[category].append(example)

        for category, examples in category_to_examples.items():
            cat_dataset = Dataset.from_list(examples)
            file_path = os.path.join(local_dir, category, f'{split_name}.parquet')
            cat_dataset.to_parquet(file_path)

            print(f"Saved {len(examples)} samples for category '{category}' in split '{split_name}'.")

            if hdfs_dir is not None:
                dst_path = os.path.join(hdfs_dir, category, f'{split_name}.parquet')
                copy(src=file_path, dst=dst_path)


    save_by_category(train_dataset, 'train')
    save_by_category(test_dataset, 'test')
    save_by_category(val_dataset, 'val')
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 