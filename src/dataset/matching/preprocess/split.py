import pandas as pd
import random
import os

def split_train_test_and_save(df, train_ratio=0.33, seed=42, save_dir="./"):
    random.seed(seed)

    # Shuffle unique query IDs
    query_ids = df['query-id'].unique().tolist()
    random.shuffle(query_ids)

    # Split query IDs into training and inductive sets
    num_queries = len(query_ids)
    num_train = int(num_queries * train_ratio)
    train_query_ids = set(query_ids[:num_train])
    inductive_query_ids = set(query_ids[num_train:])

    # Select candidate training samples
    train_candidates = df[df['query-id'].isin(train_query_ids)]

    # Balance by class in training set
    train_df_list = []
    grouped_train = train_candidates.groupby('score')
    min_class_train = grouped_train.size().min()
    for cls, group in grouped_train:
        train_df_list.append(group.sample(n=min_class_train, random_state=seed))
    train_df = pd.concat(train_df_list).reset_index(drop=True)

    # Build test candidates (excluding used training rows)
    test_candidates = df[~df.index.isin(train_df.index)]

    # Inductive = query-id not seen in training
    test_inductive_candidates = test_candidates[test_candidates['query-id'].isin(inductive_query_ids)]

    # Transductive = query-id seen in training but samples not in training set
    test_transductive_candidates = test_candidates[test_candidates['query-id'].isin(train_query_ids)]

    # Balance inductive test set
    test_inductive_list = []
    grouped_inductive = test_inductive_candidates.groupby('score')
    min_class_inductive = grouped_inductive.size().min()
    for cls, group in grouped_inductive:
        test_inductive_list.append(group.sample(n=min_class_inductive, random_state=seed))
    test_inductive_df = pd.concat(test_inductive_list).reset_index(drop=True)

    # Balance transductive test set
    test_transductive_list = []
    grouped_transductive = test_transductive_candidates.groupby('score')
    min_class_transductive = grouped_transductive.size().min()
    for cls, group in grouped_transductive:
        test_transductive_list.append(group.sample(n=min_class_transductive, random_state=seed))
    test_transductive_df = pd.concat(test_transductive_list).reset_index(drop=True)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save each split
    train_df.to_csv(f"{save_dir}/train.tsv", sep='\t', index=False)
    test_transductive_df.to_csv(f"{save_dir}/test_transductive.tsv", sep='\t', index=False)
    test_inductive_df.to_csv(f"{save_dir}/test_inductive.tsv", sep='\t', index=False)

    print(f"Saved: train={len(train_df)}, transductive={len(test_transductive_df)}, inductive={len(test_inductive_df)} at {save_dir}/")

    return train_df, test_transductive_df, test_inductive_df




if __name__ == '__main__':
    df = pd.read_csv("/shared/eng/jl254/server-05/code/TinyZero/data/raw_data/trialgpt/raw/trec_2021/qrels/test.tsv", sep='\t')
    print(f'Total: {len(df)}')
    train, trans, inductive = split_train_test_and_save(df, train_ratio=0.8, save_dir="data/raw_data/matching")