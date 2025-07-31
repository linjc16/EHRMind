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

def evaluate_seen_unseen(json_file_path, seen_unseen_path):
    # Load prediction result file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Load seen/unseen mapping
    with open(seen_unseen_path, 'r') as f:
        seen_unseen_dict = json.load(f)

    # Build a reverse lookup: id -> (category, seen/unseen)
    id_category_status = {}
    for category, d in seen_unseen_dict.items():
        for idx in d['seen']:
            id_category_status[str(idx)] = (category, 'seen')
        for idx in d['unseen']:
            id_category_status[str(idx)] = (category, 'unseen')

    # Store scores
    scores = defaultdict(lambda: defaultdict(list))  # scores[category][seen/unseen]

    for id_str, item in data.items():
        category, status = id_category_status.get(id_str, (None, None))

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
            scores[category][status].append(score)
        except Exception as e:
            scores[category][status].append(0)
            print(f"Error processing ID {id_str}: {e}")
            continue

    # Compute accuracy
    results = {}
    for category in scores:
        results[category] = {}
        for status in ['seen', 'unseen']:
            vals = scores[category][status]
            results[category][status] = sum(vals) / len(vals) if vals else 0.0

    # Compute global averages
    all_seen = [v for cat in scores.values() for v in cat['seen']]
    all_unseen = [v for cat in scores.values() for v in cat['unseen']]

    overall = {
        'seen': sum(all_seen) / len(all_seen) if all_seen else 0.0,
        'unseen': sum(all_unseen) / len(all_unseen) if all_unseen else 0.0,
    }

    print("Per-category accuracy:")
    for cat, v in results.items():
        print(f"  {cat}: seen = {v.get('seen', 0):.4f}, unseen = {v.get('unseen', 0):.4f}")
    print(f"\nOverall: seen = {overall['seen']:.4f}, unseen = {overall['unseen']:.4f}")

    return results, overall


if __name__ == '__main__':
    # json_file_path = 'results/medcalc/eval_results_medcalc-llama3.2-3b.json'
    # json_file_path = 'results/medcalc/eval_results_medcalc-sft-p20-llama3.2-3b.json'
    # json_file_path = 'results/medcalc/llama3.2-3b-rft.json'
    # json_file_path = 'results/medcalc/llama3.2-3b-sft-rft.json'
    json_file_path = 'results/medcalc/o3-mini.json'
    # json_file_path = 'results/medcalc/deepseek-r1.json'
    seen_unseen_path = 'src/eval/medcalc/analysis/category_seen_unseen.json'
    evaluate_seen_unseen(json_file_path, seen_unseen_path)
