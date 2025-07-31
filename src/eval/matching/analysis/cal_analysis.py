import json
import re
import math
from collections import defaultdict, Counter
import sys
sys.path.append('./')

import pdb

def extract_answer_json(text):
    text += '</answer>'
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            print("JSON decode failed:", repr(match.group(1)))  
    
    fallback_match = re.search(r'"idx"\s*:\s*"?([\d\s.]+)"?', text)
    if fallback_match:
        chosen_index = int(fallback_match.group(1))
        return {"idx": chosen_index}
    
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

def evaluate_pass_k(json_file_path, k=1, pass_threshold=5, entropy_threshold=1.0):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    category_scores = defaultdict(list)
    category_entropy = defaultdict(list)
    category_confident_pass = defaultdict(list)

    for item in data.values():
        outputs = item.get("outputs", [])[:k]
        answer_counts = Counter()

        num_correct = 0
        for output in outputs:
            try:
                answer_dict = extract_answer_json(output)
                answer = answer_dict.get("idx", None)
                if answer is not None:
                    answer_counts[answer] += 1
                    if answer == item['target']:
                        num_correct += 1
            except Exception as e:
                print(f"Error processing item: {e}")

        score = 1 if num_correct >= pass_threshold else 0
        entropy = compute_entropy(answer_counts)
        
        cat = item['target']
        category_scores[cat].append(score)
        if score == 1:
            category_entropy[cat].append(entropy)

        # Confident correct: correct and low entropy

        cat = item['target']
        category_scores[cat].append(score)
        # category_entropy[cat].append(entropy)

        # Confident correct: correct and low entropy
        confident_score = 1 if (score == 1 and entropy < entropy_threshold) else 0
        category_confident_pass[cat].append(confident_score)

    # Aggregation
    category_passk = {
        cat: sum(scores) / len(scores) for cat, scores in category_scores.items()
    }
    category_entropy_avg = {
        cat: sum(ents) / len(ents) for cat, ents in category_entropy.items()
    }
    category_confident_passk = {
        cat: sum(scores) / len(scores) for cat, scores in category_confident_pass.items()
    }

    all_scores = [s for scores in category_scores.values() for s in scores]
    all_entropy = [e for ents in category_entropy.values() for e in ents]
    all_confident = [s for scores in category_confident_pass.values() for s in scores]

    overall_passk = sum(all_scores) / len(all_scores) if all_scores else 0
    overall_entropy = sum(all_entropy) / len(all_entropy) if all_entropy else 0
    overall_confident_passk = sum(all_confident) / len(all_confident) if all_confident else 0

    # Print
    print(f"\nCategory pass@{k} (≥{pass_threshold} correct): {category_passk}")
    print(f"Category entropy: {category_entropy_avg}")
    print(f"Category confident pass@{k} (<{entropy_threshold}): {category_confident_passk}")
    print(f"\nOverall pass@{k}: {overall_passk:.4f}")
    print(f"Overall entropy: {overall_entropy:.4f}")
    print(f"Overall confident pass@{k}: {overall_confident_passk:.4f}")
    
    return category_passk, overall_passk, category_entropy_avg, overall_entropy, category_confident_passk, overall_confident_passk


if __name__ == '__main__':
    # json_file_path = 'results/matching/analysis/llama3.2-3b-500_pass12.json'
    json_file_path = 'results/matching/analysis/llama3.2-3b-sft-100_pass12.json'

    k = 12
    pass_threshold = math.ceil(k // 3) + 2
    entropy_threshold = math.log(3) * 0.8
    evaluate_pass_k(json_file_path, k=k, pass_threshold=pass_threshold, entropy_threshold=entropy_threshold)
