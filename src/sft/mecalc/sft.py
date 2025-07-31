from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=['full', '10p', '15p', '20p'], default='full')
args = parser.parse_args()

mode = args.mode

if mode == 'full':
    train_file_path = 'data/local_index_search/medcalc/sft/train.parquet'
    val_file_path = 'data/local_index_search/medcalc/sft/val.parquet'
    output_dir = "./checkpoints/medcalc-sft-full"
elif mode == '10p':
    train_file_path = 'data/local_index_search/medcalc/sft_10p/train.parquet'
    val_file_path = 'data/local_index_search/medcalc/sft_10p/val.parquet'
    output_dir = "./checkpoints/medcalc-sft-10p"
elif mode == '15p':
    train_file_path = 'data/local_index_search/medcalc/sft_15p/train.parquet'
    val_file_path = 'data/local_index_search/medcalc/sft_15p/val.parquet'
    output_dir = "./checkpoints/medcalc-sft-15p"
elif mode == '20p':
    train_file_path = 'data/local_index_search/medcalc/sft_20p/train.parquet'
    val_file_path = 'data/local_index_search/medcalc/sft_20p/val.parquet'
    output_dir = "./checkpoints/medcalc-sft-20p"

model_name = "/shared/eng/jl254/server-05/code/TinyZero/models/llama3"
cache_dir = "/srv/local/data/linjc/hub"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir,
    attn_implementation="flash_attention_2"
)

tokenizer.pad_token_id = tokenizer.eos_token_id


dataset = load_dataset("parquet", data_files={
    "train": train_file_path,
    "validation": val_file_path
})

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    args=training_args,
    max_seq_length=4096
)

trainer.train()
