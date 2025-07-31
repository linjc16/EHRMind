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
import pandas as pd
import pdb



PROMPT = """You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the eligibility criteria of a clinical trial to determine the patient's eligibility.

The assessment of eligibility has a three-point scale: 
0) Irrelevant (patient does not have sufficient information to qualify for the trial); 
1) Excluded (patient meets inclusion criteria, but is excluded on the grounds of the trial's exclusion criteria); and 
2) Eligible (patient meets inclusion criteria and exclusion criteria do not apply).
You should make a trial-level eligibility on each patient for the clinical trial, i.e., output the scale for the assessment of eligibility. 

Here is the patient note:
{note}

Here is the clinial trial:
{trial}
-----------------"""

def make_prefix(dp):
    input_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n""" + PROMPT.format(note=dp['note'], trial=dp['trial'])
    input_str += """\nPlease show your entire reasoning process in **a single** <think> </think> block (do not open or close the tag more than once). Your final response must be in JSON format within <answer> </answer> tags. For example,
<think>
[entire reasoning process here, you should analyze each criterion one by one]
</think>
<answer>
{
    "prediction": str(one_of_["Irrelevant", "Excluded", "Eligible"]),
    "idx": int(matching_index_of_this_data_point)
}
</answer>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\nLet me solve this step by step.\n<think>"""

    return input_str

def load_queries():
    queries_dict = {}
    
    with open("/shared/eng/jl254/server-05/code/TinyZero/data/raw_data/trialgpt/raw/trec_2021/queries.jsonl", 'r') as f:
        for line in f:
            query = json.loads(line)
            queries_dict[query['_id']] = query['text']

    return queries_dict

def get_trial_text(data):
    text = f"""Title: {data['brief_title'].strip()}
Inclusion Criteria: 
```
{data['inclusion_criteria'].strip()}
```
Exclusion Criteria: 
```
{data['exclusion_criteria'].strip()}
```"""
    
    return text

def build_data_samples(df, queries_dict, trial_info):
    samples = []

    for _, row in df.iterrows():
        query_id = row['query-id']
        corpus_id = row['corpus-id']
        label = row['score']

        # Get note and trial text
        note = queries_dict.get(query_id, "")
        trial_data = trial_info.get(corpus_id, {})
        try:
            trial_text = get_trial_text(trial_data)
        except:
            print(f"Error getting trial text for corpus_id: {corpus_id}")
            continue

        sample = {
            'note': note,
            'trial': trial_text,
            'label': label
        }

        samples.append(sample)

    return samples

def load_matching_dataset():
    qrel_train = pd.read_csv("data/raw_data/matching/train.tsv", sep='\t')
    
    # random select 1000 samples as the validation set
    qrel_val = qrel_train.sample(n=1000, random_state=42)
    qrel_train = qrel_train.drop(qrel_val.index)

    queries_dict = load_queries()

    with open('/shared/eng/jl254/server-05/code/TinyZero/data/raw_data/trialgpt/trial_info.json', 'r') as f:
        trial_info = json.load(f)
    
    train_data = build_data_samples(qrel_train, queries_dict, trial_info)
    val_data = build_data_samples(qrel_val, queries_dict, trial_info)

    qrel_test_trans = pd.read_csv("data/raw_data/matching/test_transductive.tsv", sep='\t')
    test_data_trans = build_data_samples(qrel_test_trans, queries_dict, trial_info)

    qrel_test_ind = pd.read_csv("data/raw_data/matching/test_inductive.tsv", sep='\t')
    test_data_ind = build_data_samples(qrel_test_ind, queries_dict, trial_info)

    return train_data, val_data, test_data_trans, test_data_ind


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/local_index_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset', type=str, default='matching')

    args = parser.parse_args()
    
    data_source = args.dataset
    
    train_data, val_data, test_data_trans, test_data_ind = load_matching_dataset()

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset_trans = Dataset.from_list(test_data_trans)
    test_dataset_ind = Dataset.from_list(test_data_ind)
    
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = make_prefix(example)
            solution = {
                "target": example['label'],
            }
            
            data = {
                "data_source": data_source + '_' + split,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "matching",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val', data_source), with_indices=True)
    test_dataset_trans = test_dataset_trans.map(function=make_map_fn('test', data_source + '_transductive'), with_indices=True)
    test_dataset_ind = test_dataset_ind.map(function=make_map_fn('test', data_source + '_inductive'), with_indices=True)
    
    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)

    val_dataset = concatenate_datasets([val_dataset, test_dataset_trans, test_dataset_ind])

    test_dataset = concatenate_datasets([test_dataset_trans, test_dataset_ind])
    
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

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 