import pandas as pd
import numpy as np
import os
import json
from datasets import load_dataset, Dataset
import pdb
import argparse
import sys
sys.path.append('./')


def construct_data(args):
    data_path = os.path.join(f'data/raw_data/matching/sft/sft_samples_matching.json')
    data = load_dataset('json', data_files=data_path, split='train')

    df = data.to_pandas()

    dataset = Dataset.from_pandas(df)

    # get text field by concatenate 'input' and 'output'
    dataset = dataset.map(lambda x: {'text': x['input'] + '\n' + x['output']})

    # split data into train and val
    train_data = dataset.train_test_split(test_size=100)
    
    val_data = train_data['test']
    train_data = train_data['train']
    
    save_dir = f'data/local_index_search/matching/sft'
    os.makedirs(save_dir, exist_ok=True)
    train_data.to_parquet(f'{save_dir}/train.parquet')
    val_data.to_parquet(f'{save_dir}/val.parquet')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    construct_data(args)