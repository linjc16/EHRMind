import re
import random
import numpy as np
import ast
import operator
import pdb
import json
from collections import defaultdict
import sys
import math
import os
sys.path.append('./')


DATA_COUNT = {
    'new_acutemi':{
        '0': 1958,
        '1': 144,
    },
    'new_hyperlipidemia': {
        '0': 1123,
        '1': 170
    },
    'new_hypertension': {
        '0': 1078,
        '1': 155
    },
    'new_pancan': {
        '0': 2140,
        '1': 55
    }
}

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1].strip()
    else:
        print("[Error] Failed to locate model response header")
        return None, processed_str

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str
        

def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # processed_str = '<think> </think>' + processed_str
    
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")
    
    return validation_passed


def check_json_format(json_str, do_print=False):
    """Check if the given string is a valid JSON and follows the expected structure."""
    try:
        if not json_str:
            if do_print:
                print("[Error] Empty JSON string")
            return False
        
        data = json.loads(json_str)

        # Required keys
        required_keys = {"prediction", "label"}
        if not all(key in data for key in required_keys):
            if do_print:
                print("[Error] Missing required keys in JSON")
            return False

        # # Validate label value
        # label = data["label"]
        # prediction = data["prediction"]
        
        # if label == 0:
        #     if prediction != label_name:
        #         if do_print:
        #             print(f"[Error] For label 0, prediction must be '{label_name}'")
        #         return False
        # elif label == 1:
        #     if prediction != label_name:
        #         if do_print:
        #             print(f"[Error] For label 1, prediction must be '{label_name}'")
        #         return False
        # else:
        #     if do_print:
        #         print("[Error] Label must be 0 or 1")
        #     return False

        return True
    except json.JSONDecodeError:
        if do_print:
            print("[Error] JSON decoding failed")
        return False

def check_tail_after_answer_tag(processed_str: str, do_print=False) -> bool:
    """
    Check whether there is only whitespace or allowed special tokens after </answer>.
    Returns True if clean, False if anything unexpected is found.
    """
    ALLOWED_TOKENS = {"<|eot_id|>"}

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        if do_print:
            print("[Error] No <answer>...</answer> block found")
        return False

    last_match = matches[-1]
    tail = processed_str[last_match.end():].strip()

    if not tail:
        return True
    
    # Allow tail if it consists only of allowed tokens
    tokens = tail.split()
    if all(token in ALLOWED_TOKENS for token in tokens):
        return True

    if do_print:
        print(f"[Error] Unexpected content after </answer>: {repr(tail)}")
    return False



def calculate_answer_score(json_str, label, label_name, data_split, dataset_name):
    """Calculate answer score based on final_prediction idx."""
    try:
        data = json.loads(json_str)
        pred_number = str(data.get("label"))
        target = str(label)
        prediction_name = str(data.get("prediction"))
        
        if target == pred_number and prediction_name == label_name:
            if data_split == 'train':
                answer_score = 1
            else:
                total_count = DATA_COUNT[dataset_name]['0'] + DATA_COUNT[dataset_name]['1']
                answer_score = 0.5 * total_count / DATA_COUNT[dataset_name][target]
        else:
            answer_score = 0
    
    except:
        print("[Error] Error in evaluation")
        answer_score = -2
    
    return answer_score

def compute_score(solution_str, ground_truth, data_source, format_reward=0.1):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    label = ground_truth['target']
    label_name = ground_truth['prediction']
    
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 32) == 1
    
    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    json_format_correct = check_json_format(answer_text, do_print)
    tail_clean = check_tail_after_answer_tag(processed_str, do_print)
    format_correct = response_format_correct and json_format_correct and tail_clean
    
    format_score = format_reward if format_correct else -2
    # if do_print:
    #     print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    #     print(f"Format score: {format_score}")
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Target: {label} |")
    
    if 'new_acutemi' in data_source:
        dataset_name = 'new_acutemi'
    elif 'new_hyperlipidemia' in data_source:
        dataset_name = 'new_hyperlipidemia'
    elif 'new_hypertension' in data_source:
        dataset_name = 'new_hypertension'
    elif 'new_pancan' in data_source:
        dataset_name = 'new_pancan'

    if '_test' in data_source:
        data_split = 'test'
    else:
        data_split = 'train'
    
    answer_score = 0
    if format_correct and answer_text:
        answer_score = calculate_answer_score(answer_text, label, label_name, data_split, dataset_name)
    
    if answer_score > 0:
        total_score = format_score + answer_score
    else:
        if format_score > 0:
            total_score = 0
        else:
            total_score = format_score
    
    if do_print:
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return total_score


if __name__ == '__main__':
    solution_str = """<|start_header_id|>assistant<|end_header_id|>: <think> </think> <answer>{"prediction": "no readmission", "label": 0}</answer><|eot_id|>
"""
    ground_truth = {'target': 0}
    scores = compute_score(solution_str, ground_truth, data_source='sigir_train')
    print(scores)