import json
import re
from collections import defaultdict
import sys
sys.path.append('./')

from src.eval.utils import check_correctness
import pdb

import argparse


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

def evaluate_dataset(json_file_path):
    """
    Evaluate correctness of answers in a dataset and compute category-wise and overall averages.

    Parameters:
    - json_file_path (str): Path to the JSON data file.

    Returns:
    - category_averages (dict): Average correctness per category.
    - overall_average (float): Overall average correctness.
    """
    # Load dataset
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    category_scores = defaultdict(list)

    # Iterate through each sample in the dataset
    for item in data.values():
        try:
            # Parse the JSON from the extracted answer
            answer_dict = extract_answer_json(item.get("generated_text", ""))
            answer = str(answer_dict.get("answer", ""))
            
            # Compute correctness using the user-defined function
            score = check_correctness(
                answer=answer,
                ground_truth=item['target'],
                calid=item['calid'],
                upper_limit=item['upper_limit'],
                lower_limit=item['lower_limit']
            )
            category_scores[item['category']].append(score)
        except Exception as e:
            category_scores[item['category']].append(0)
            print(f"Error processing item: {e}")

    # Compute per-category averages
    category_averages = {
        cat: sum(scores) / len(scores) for cat, scores in category_scores.items()
    }

    # Compute overall average
    all_scores = [score for scores in category_scores.values() for score in scores]
    overall_average = sum(all_scores) / len(all_scores) if all_scores else 0

    print(f"Category averages: {category_averages}")
    print(f"Overall average: {overall_average}")
    
    return category_averages, overall_average

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, default='results/medcalc/eval_results_medcalc-sft-full-llama3.2-3b.json')
    args = parser.parse_args()
    # json_file_path = 'results/medcalc/eval_results_medcalc-sft-full-llama3.2-3b.json'
    # json_file_path = 'results/medcalc/eval_results_medcalc-sft-p20-llama3.2-3b.json'
    # json_file_path = 'results/medcalc/eval_results_medcalc-llama3.2-3b.json'
    # json_file_path = 'results/medcalc/deepseek-r1.json'
    # json_file_path = 'results/medcalc/o3-mini.json'
    # json_file_path = 'results/medcalc/claude-haiku.json'
    # json_file_path = 'results/medcalc/claude-sonnet.json'

    json_file_path = args.json_file_path
    evaluate_dataset(json_file_path)