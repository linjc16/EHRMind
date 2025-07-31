import pandas as pd
import numpy as np
import os
import json
from datasets import load_dataset, Dataset
import pdb
import argparse
import sys
sys.path.append('./')

from src.dataset.medcalc.utils import get_balanced_validation_set, count_samples_per_category


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
</answer>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\nLet me solve this step by step.\n<think>"""

    answer_json = json.dumps({"answer": dp["Ground Truth Answer"]}, ensure_ascii=False)
    input_str += f"""
{dp['Ground Truth Explanation']}
</think>
<answer>
{answer_json}
</answer><|eot_id|>"""

    return input_str



def construct_data(args):
    data_path = os.path.join(f'data/raw_data/medcalc/train_data.csv')
    data = load_dataset('csv', data_files=data_path, split='train')

    # add one column text = make_prefix + output
    data = data.map(lambda x: {'text': make_prefix(x)})
    df = data.to_pandas()

    PERCENTAGE_MAP = {
        '5p': 0.05,
        '10p': 0.10,
        '15p': 0.15,
        '20p': 0.20,
        'full': 1.0
    }


    # Define categories by priority
    base_minorities = ['diagnosis', 'severity', 'date', 'dosage conversion']
    mid_priority = ['risk']
    majorities = ['physical', 'lab test']

    target_pct = PERCENTAGE_MAP[args.mode]
    total_target_count = int(len(df) * target_pct)

    CATEGORY_CAP = {
        'diagnosis': total_target_count // 7,
        'severity': total_target_count // 7,
        'date': total_target_count // 7,
        'dosage conversion': total_target_count // 7,
        'lab test': total_target_count // 7,
        'physical': total_target_count // 7,
    }

    # Step 1: include all base minorities
    minority_samples = []

    for cat in base_minorities:
        cat_df = df[df['Category'] == cat]
        cap = min(CATEGORY_CAP.get(cat, len(cat_df)), len(cat_df))  # default to full if no cap
        sampled = cat_df if len(cat_df) <= cap else cat_df.sample(n=cap, random_state=42)
        minority_samples.append(sampled)

    base_minorities_df = pd.concat(minority_samples)
    base_count = len(base_minorities_df)

    # Step 2: add risk if space allows
    risk_df = df[df['Category'].isin(mid_priority)]
    remaining_count = total_target_count - base_count
    sampled_risk_df = risk_df.sample(n=min(remaining_count - (total_target_count // 7) * 2, len(risk_df)), random_state=42)
    risk_count = len(sampled_risk_df)

    # Step 3: add majorities if space still remains
    remaining_count -= risk_count
    majorities_df = df[df['Category'].isin(majorities)]
    
    majority_samples = []

    for cat in majorities:
        cat_df = df[df['Category'] == cat]
        cap = min(CATEGORY_CAP.get(cat, len(cat_df)), len(cat_df))  # default to full if no cap
        sampled = cat_df if len(cat_df) <= cap else cat_df.sample(n=cap, random_state=42)
        majority_samples.append(sampled)
    
    sampled_majority_df = pd.concat(majority_samples)

    # Combine all
    final_df = pd.concat([base_minorities_df, sampled_risk_df, sampled_majority_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = Dataset.from_pandas(final_df)

    # Split validation set
    val_data, train_data = get_balanced_validation_set(dataset, 'Category', total_val_size=100)
    
    print(count_samples_per_category(train_data, 'Category'))

    # Save
    save_dir = f'data/local_index_search/medcalc/sft_{args.mode}'
    os.makedirs(save_dir, exist_ok=True)
    train_data.to_parquet(f'{save_dir}/train.parquet')
    val_data.to_parquet(f'{save_dir}/val.parquet')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['full', '10p', '15p', '20p'], default='full')
    args = parser.parse_args()
    construct_data(args)