import pandas as pd
import json

# 分类合并映射
CATEGORY_MERGE_MAP = {
    "lab test": "lab",
    "lab": "lab",
    "dosage conversion": "dosage",
    "dosage": "dosage",
}

train = pd.read_parquet("data/local_index_search/medcalc/train.parquet")
test = pd.read_parquet("data/local_index_search/medcalc/test.parquet")
# val = pd.read_parquet("data/local_index_search/medcalc/val.parquet")

# val_filtered = val[~val['data_source'].str.contains("_test", na=False)]
# train = pd.concat([train, val_filtered], ignore_index=True)

train['Category'] = train['Category'].map(CATEGORY_MERGE_MAP).fillna(train['Category'])
test['Category'] = test['Category'].map(CATEGORY_MERGE_MAP).fillna(test['Category'])

train['Question'] = train['Question'].str.replace("’", "'", regex=False)
test['Question'] = test['Question'].str.replace("’", "'", regex=False)

test['global_id'] = range(len(test))

result = {}


categories = test['Category'].unique()

for category in categories:
    train_questions = set(train[train['Category'] == category]['Question'])
    test_subset = test[test['Category'] == category]

    seen_ids = []
    unseen_ids = []
    seen_questions = set()
    unseen_questions = set()

    for _, row in test_subset.iterrows():
        if row['Question'] in train_questions:
            seen_ids.append(row['global_id'])
            seen_questions.add(row['Question'])
        else:
            unseen_ids.append(row['global_id'])
            unseen_questions.add(row['Question'])

    result[category] = {
        'seen': seen_ids,
        'unseen': unseen_ids,
        'seen_questions': sorted(seen_questions),
        'unseen_questions': sorted(unseen_questions),
        'seen_count': len(seen_questions),
        'unseen_count': len(unseen_questions)
    }

with open("src/eval/medcalc/analysis/category_seen_unseen.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
