import json
import re
import os
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd
import argparse
import pdb

failure_type_map = {
    "0": "Excluded",
    "1": "Irrelevant",
    "2": "Eligible",
}

def extract_answer_json(text):
    """Extract JSON dict inside <answer>...</answer> tag or fallback to extract 'chosen_index'."""
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass 
    
    fallback_match = re.search(r'"idx"\s*:\s*(\d+)', text)
    if fallback_match:
        chosen_index = int(fallback_match.group(1))
        return {"idx": chosen_index}
    
    return {}

def evaluate_predictions(json_file_path, output_path):
    # Load JSON data from file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels

    for item in data.values():
        true_label = item.get("label")
        answer_dict = extract_answer_json(item.get("output", ""))
        pred_label = answer_dict.get("idx", "None")

        if true_label is None:
            true_label = "None"

        y_true.append(str(true_label))
        y_pred.append(str(pred_label))
    
    # Calculate overall metrics
    results = {
        "overall": {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        },
        "per_class": {}
    }
    
    # Calculate per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for label, metrics in report.items():
        if label.isdigit() or label == "None":
            if label not in failure_type_map:
                continue
            class_name = failure_type_map.get(label, f"Unknown_{label}")
            results["per_class"][class_name] = {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"]
            }
    
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print("\nConfusion Matrix:\n")
    print(cm_df.to_string())

    # Write results to output JSON file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)

    return results


if __name__ == '__main__':
    model_names = [
        # 'gpt-4o',
        # 'claude-sonnet',
        # 'claude-haiku',
        # 'o3-mini',
        # 'deepseek-r1',
        # 'llama-3-8b',
        # 'llama-3-70b',
        # 'llama3.2-3b',
        # 'llama3.2-3b-sft',
        'llama3.2-3b-rft',
        'llama3.2-3b-sft-rft'
    ]
    

    for model_name in model_names:
        input_path = f'results/matching/output/{model_name}.json'
        output_path = f'results/matching/metrics/{model_name}.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Evaluating {model_name}...")
        evaluate_predictions(input_path, output_path)
