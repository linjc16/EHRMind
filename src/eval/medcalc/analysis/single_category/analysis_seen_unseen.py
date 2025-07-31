import json
import re
from collections import defaultdict
import sys
sys.path.append('./')

from src.eval.utils import check_correctness
import pdb

def extract_answer_json(text):
    """Extract JSON dict inside <answer>...</answer> tag or fallback to extract 'answer'."""
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            print("JSON decode failed:", repr(match.group(1)))  
    
    fallback_match = re.search(r'"answer"\s*:\s*"?([\d\s.]+)"?', text)
    if fallback_match:
        chosen_index = int(fallback_match.group(1))
        return {"answer": chosen_index}
    
    return {}

def evaluate_seen_unseen(json_file_path, seen_unseen_path, target_category):
    # Load prediction result file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Load seen/unseen mapping
    with open(seen_unseen_path, 'r') as f:
        seen_unseen_dict = json.load(f)

    # Check if the category exists
    if target_category not in seen_unseen_dict:
        raise ValueError(f"Category '{target_category}' not found in seen/unseen file.")

    # Build reverse lookup only for the target category
    id_category_status = {}
    d = seen_unseen_dict[target_category]
    for idx in d['seen']:
        id_category_status[str(idx)] = 'seen'
    for idx in d['unseen']:
        id_category_status[str(idx)] = 'unseen'

    # Store scores
    scores = {'seen': [], 'unseen': []}

    for id_str, item in data.items():
        status = id_category_status.get(id_str)
        try:
            answer_dict = extract_answer_json(item.get("generated_text", ""))
            answer = str(answer_dict.get("answer", ""))

            score = check_correctness(
                answer=answer,
                ground_truth=item['target'],
                calid=item['calid'],
                upper_limit=item['upper_limit'],
                lower_limit=item['lower_limit']
            )
            scores[status].append(score)
        except Exception as e:
            scores[status].append(0)
            print(f"Error processing ID {id_str}: {e}")
            continue

    # Compute accuracy
    results = {}
    for status in ['seen', 'unseen']:
        vals = scores[status]
        results[status] = sum(vals) / len(vals) if vals else 0.0

    print(f"\nAccuracy for category '{target_category}':")
    print(f"  seen = {results.get('seen', 0):.4f}, unseen = {results.get('unseen', 0):.4f}")

    return results


if __name__ == '__main__':
    category = 'severity'
    # json_file_path = f'results/medcalc/single_category/{category}/llama3.2-3b-rft.json'
    json_file_path = f'results/medcalc/single_category/{category}/llama3.2-3b-sft-rft.json'

    seen_unseen_path = 'src/eval/medcalc/analysis/single_category/category_seen_unseen_sc.json'
    evaluate_seen_unseen(json_file_path, seen_unseen_path, category)
    