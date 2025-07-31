import json
import re
from collections import defaultdict, Counter
import sys
import math
sys.path.append('./')

from src.eval.utils import check_correctness
import pdb


def extract_answer_json(text):
    text += '</answer>'
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

def compute_entropy(dist: Counter):
    """Compute entropy from a Counter distribution."""
    total = sum(dist.values())

    if total == 0:
        return 0.0
    entropy = 0.0
    for count in dist.values():
        p = count / total
        entropy -= p * math.log(p + 1e-8)
    return entropy

def evaluate_pass_k(json_file_path, k=1):
    """
    Evaluate pass@k for a dataset.

    Parameters:
    - json_file_path (str): Path to the JSON data file.
    - k (int): Value of k in pass@k.

    Returns:
    - category_passk (dict): pass@k per category.
    - overall_passk (float): Overall pass@k.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    category_scores = defaultdict(list)

    for item in data.values():
        outputs = item.get("outputs", [])[:k]
        found_correct = False

        if item['category'] not in ['diagnosis', 'severity', 'risk']:
            for output in outputs:
                try:
                    answer_dict = extract_answer_json(output)
                    answer = str(answer_dict.get("answer", ""))
                    if check_correctness(
                        answer=answer,
                        ground_truth=item['target'],
                        calid=item['calid'],
                        upper_limit=item['upper_limit'],
                        lower_limit=item['lower_limit']
                    ):
                        found_correct = True
                        break  # At least one correct answer in top-k
                except Exception as e:
                    print(f"Error processing item: {e}")
            
            score = 1 if found_correct else 0
        else:
            answer_counts = Counter()
            num_correct = 0
            for output in outputs:
                try:
                    answer_dict = extract_answer_json(output)
                    answer = str(answer_dict.get("answer", None))
                    if answer is not None:
                        answer_counts[answer] += 1
                        if answer == item['target']:
                            num_correct += 1
                except Exception as e:
                    print(f"Error processing item: {e}")
            
            entropy = compute_entropy(answer_counts)
            pass_threshold = math.ceil(k // 7) + 1 + 1 if item['category'] in ['diagnosis', 'severity'] else math.ceil(k // 21) + 1 + 1
            entropy_threshold = math.log(7) * 0.8 if item['category'] in ['diagnosis', 'severity'] else math.log(21) * 0.8
            
            score = 1 if (num_correct >= pass_threshold and entropy < entropy_threshold) else 0

        category_scores[item['category']].append(score)

    # Compute category-level pass@k
    category_passk = {
        cat: sum(scores) / len(scores) for cat, scores in category_scores.items()
    }

    all_scores = [score for scores in category_scores.values() for score in scores]
    overall_passk = sum(all_scores) / len(all_scores) if all_scores else 0

    # filtered = {
    #     cat: scores for cat, scores in category_scores.items()
    #     if not (all(s == 0 for s in scores) or all(s == 1 for s in scores))
    # }
    # filtered_passk = {
    #     cat: sum(scores) / len(scores) for cat, scores in filtered.items()
    # }
    # filtered_overall = (
    #     sum([sum(scores) for scores in filtered.values()]) /
    #     sum([len(scores) for scores in filtered.values()])
    #     if filtered else 0
    # )

    print(f"Category pass@{k}: {category_passk}")
    print(f"Overall pass@{k}: {overall_passk}")
    # print(f"\nFiltered (excluding all-0 and all-1):")
    # print(f"Filtered category pass@{k}: {filtered_passk}")
    # print(f"Filtered overall pass@{k}: {filtered_overall}")

    return category_passk, overall_passk#, filtered_passk, filtered_overall

if __name__ == '__main__':
    # json_file_path = 'results/medcalc/analysis/llama3.2-3b-500_pass12.json'
    json_file_path = 'results/medcalc/analysis/llama3.2-3b-sft-500_pass12.json'
    k = 12  # You can change this to pass@1, pass@3, pass@5, etc.
    evaluate_pass_k(json_file_path, k=k)
