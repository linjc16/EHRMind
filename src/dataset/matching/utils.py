import pandas as pd
import random
import json

def sample_balanced_triplet(df, label_col='score', n_total=3000, seed=42):
    """
    Sample a balanced dataset from a 3-class label column.
    
    Parameters:
    - df: pandas DataFrame with a column like 'score' containing labels 0/1/2
    - label_col: the column name for the label
    - n_total: total number of samples (must be divisible by 3)
    - seed: random seed for reproducibility
    
    Returns:
    - pd.DataFrame with balanced samples
    """
    assert n_total % 3 == 0, "Total sample count must be divisible by 3"
    n_per_class = n_total // 3
    random.seed(seed)

    sampled = []
    for label in [0, 1, 2]:
        group = df[df[label_col] == label]
        if len(group) < n_per_class:
            raise ValueError(f"Not enough samples for class {label}: only {len(group)} available.")
        sampled_group = group.sample(n=n_per_class, random_state=seed)
        sampled.append(sampled_group)

    balanced_df = pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return balanced_df



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
            'label': label,
            'query_id': query_id,
            'corpus_id': corpus_id
        }

        samples.append(sample)

    return samples