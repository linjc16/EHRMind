import argparse
import pandas as pd
from tqdm import tqdm
import json
from vllm import LLM, SamplingParams
import os

CACHE_DIR = "/srv/local/data/linjc/hub"

def evaluate_model(model_path, data_path, model_name, save_dir, batch_size=8, max_tokens=2048, pass_k=12, category_sample_size=None):
    df = pd.read_parquet(data_path)

    if category_sample_size is not None:
        print('Sampling from each category')
        df = df.groupby("Category").apply(lambda x: x.sample(n=min(len(x), category_sample_size), random_state=42)).reset_index(drop=True)
    
    print(f"Loaded {len(df)} samples from {data_path}")

    prompts = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['Ground Truth Answer'].tolist()
    calids = df['Calculator ID'].tolist()
    upper_limits = df['Upper Limit'].tolist()
    lower_limit = df['Lower Limit'].tolist()
    categories = df['Category'].tolist()

    os.makedirs(save_dir, exist_ok=True)

    llm = LLM(model=model_path, dtype="bfloat16", tokenizer=model_path)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6,
        n=pass_k,
        top_p=0.95,
        stop=["</answer>"]
    )

    generated_texts = {}
    count = 0

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        outputs = llm.generate(batch_prompts, sampling_params)

        for i, output in enumerate(outputs):
            completions = []
            for gen in output.outputs:
                text = gen.text
                if "\nassistant\n" in text:
                    text = text.split("\nassistant\n", 1)[1]
                completions.append(text.strip())

            idx = batch_start + i
            generated_texts[idx] = {
                "outputs": completions,  # This now stores 20 completions
                "target": targets[idx],
                'calid': calids[idx],
                'upper_limit': upper_limits[idx],
                'lower_limit': lower_limit[idx],
                'category': categories[idx]
            }
            
            if count % 100 == 0:
                with open(os.path.join(save_dir, f"{model_name}_pass{pass_k}.json"), "w") as f:
                    json.dump(generated_texts, f, indent=4)

            count += 1
    
    # Final save
    with open(os.path.join(save_dir, f"{model_name}_pass{pass_k}.json"), "w") as f:
        json.dump(generated_texts, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/Panacea-Zero/matching-qwen2.5-3b-inst-ppo-2gpus/actor/global_step_400")
    parser.add_argument("--data_path", type=str, default="data/local_index_search/medcalc/test.parquet")
    parser.add_argument("--model_name", type=str, default="llama3-3b")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--category_sample_size", type=int, default=None, help="Maximum samples per category (optional)")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    evaluate_model(args.model_path, args.data_path, args.model_name, args.save_dir, args.batch_size, category_sample_size=args.category_sample_size)

if __name__ == "__main__":
    main()
