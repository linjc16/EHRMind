import pickle
import json
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('./')
from src.utils.gpt_azure import gpt_chat_4o
from src.dataset.matching.utils import sample_balanced_triplet, load_queries, build_data_samples

import pdb

label_to_text_dict = {
    0: 'Irrelevant',
    1: 'Excluded',
    2: 'Eligible'

}

TASKS = {
    "trial_matching": {
        "description": """
Patient-Trial Matching Task:
Objective: Determine whether a patient is eligible for a given clinical trial based on the patient's medical note and the trial's inclusion/exclusion criteria.

Labels:
0) Irrelevant (patient does not have sufficient information to qualify for the trial); 
1) Excluded (patient meets inclusion criteria, but is excluded on the grounds of the trial's exclusion criteria); and 
2) Eligible (patient meets inclusion criteria and exclusion criteria do not apply).

Key Considerations:
- Carefully **evaluate each inclusion and exclusion criterion individually**.
- For each criterion, determine whether the patient **clearly satisfies**, **clearly violates**, or has **insufficient information**.
"""
    }
}

TASKS_ABBR = {
    "trial_matching": {
        "description": """
You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the eligibility criteria of a clinical trial to determine the patient's eligibility.

The assessment of eligibility has a three-point scale: 
0) Irrelevant (patient does not have sufficient information to qualify for the trial); 
1) Excluded (patient meets inclusion criteria, but is excluded on the grounds of the trial's exclusion criteria); and 
2) Eligible (patient meets inclusion criteria and exclusion criteria do not apply).
You should make a trial-level eligibility on each patient for the clinical trial, i.e., output the scale for the assessment of eligibility. 
"""
    }
}


def generate_reasoning(patient_context, trial_info, ground_truth, task='trial_matching'):    
    prompt = f"""
Given the following task description, patient EHR context, clinical trial information, and the ground truth eligibility label, provide a step-by-step reasoning process that leads to the correct prediction:

========================================
# Task #
{TASKS[task]['description']}

========================================
# Patient EHR Note #

{patient_context}

========================================
# Clinical Trial #

{trial_info}

========================================
# Ground Truth #

{label_to_text_dict[ground_truth]}

========================================

Please provide a step-by-step reasoning process that leads to the correct prediction based on the patient's EHR context.

**The reasoning chain should follow this structured format:**

1. **Patient and Clincial Trial Overview**: Go over the key information in the patient's EHR context and the clinical trial criteria, with the **Key Considerations** from the task description in mind.
2. **Reasoning Towards Prediction**: Integrate the above information to logically reason towards the predicted outcome.
3. **Conclusion**: Summarize the reasoning and state the prediction without mentioning the ground truth label.

The reasoning should be comprehensive, medically sound, and clearly explain how the patient's information leads to the predicted outcome.

**Important Notes:**
- **Do not mention the ground truth label in the reasoning process**.
- Use the relevant knowledge as needed, but **the main focus should be on the patient's EHR context and the clinical trial**.

After generating the reasoning chain, please review it and indicate your confidence in the reasoning chain at the end.

Options of confidence: [Very Confident, Confident, Neutral, Not Confident, Very Not Confident.]

**Output Format:**

# Reasoning Chain #

1. **Patient and Clincial Trial Overview**:
   [YOUR OUTPUT]

2. **Reasoning Towards Prediction**:
   [YOUR OUTPUT]

3. **Conclusion**:
   [YOUR OUTPUT]

# Confidence #
[CONFIDENCE (choose one: "Very Confident", "Confident", "Neutral", "Not Confident", "Very Not Confident")]
"""

    retries = 0
    while retries < 5:
        try:
            # response = get_claude_response(llm="sonnet", prompt=prompt)
            response = gpt_chat_4o(prompt=prompt)

            return response
        except:
            retries += 1
            print(f"ReadTimeoutError occurred. Retrying... (Attempt {retries})")
    return response


def construct_sft_input_output(patient_context, trial_info, task, reasoning, ground_truth):
    context = patient_context

    answer_json = json.dumps({
        "prediction": label_to_text_dict[ground_truth],
        "idx": ground_truth,
        }, ensure_ascii=False)

    input_ = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>
Given the following task description and patient EHR context, please provide a step-by-step reasoning process that leads to the prediction outcome based on the patient's context.
After the reasoning process, provide the prediction.

========================================
# Task #
{TASKS_ABBR[task]['description'].strip()}

========================================
Here is the patient note:
{context}

Here is the clinial trial:
{trial_info}
-----------------
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    output_ = f"""
<think>
{reasoning}
</think>
<answer>
{answer_json}
</answer><|eot_id|>"""

    return input_.strip(), output_.strip()

if __name__ == '__main__':
    qrel_train = pd.read_csv("data/raw_data/matching/train.tsv", sep='\t')
   
    qrel_subset_data = sample_balanced_triplet(qrel_train, label_col='score', n_total=3000, seed=42)

    queries_dict = load_queries()

    with open('/shared/eng/jl254/server-05/code/TinyZero/data/raw_data/trialgpt/trial_info.json', 'r') as f:
        trial_info = json.load(f)
    
    subset_data = build_data_samples(qrel_subset_data, queries_dict, trial_info)

    results = []

    for idx, record in enumerate(tqdm(subset_data)):
      patient_context = record['note']
      ground_truth = record['label']
      trial_info = record['trial']
      reasoning_response = generate_reasoning(patient_context, trial_info, ground_truth, task='trial_matching')

      if "\n# Confidence #\nNot Confident" in reasoning_response:
          continue
      
      reasoning_clean = reasoning_response.split("\n# Confidence #\n")[0]
      reasoning_clean = reasoning_clean.replace("# Reasoning Chain #\n", "").strip()

      input_text, output_text = construct_sft_input_output(
         patient_context=patient_context,
         trial_info=trial_info,
         task='trial_matching',
         reasoning=reasoning_clean,
         ground_truth=ground_truth
      )
    
      
      results.append({
        "input": input_text, 
        "output": output_text,
        'query_id': record['query_id'],
        'corpus_id': record['corpus_id'],
         'label': record['label'],
      })

      if idx % 100 == 0:
         with open(f"data/raw_data/matching/sft/sft_samples_matching.json", "w") as f:
            json.dump(results, f, indent=4)
    
    with open(f"data/raw_data/matching/sft/sft_samples_matching.json", "w") as f:
      json.dump(results, f, indent=4)