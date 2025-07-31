MODEL_PATH=checkpoints/medcalc-sft-20p/checkpoint-238
DATA_PATH=data/local_index_search/medcalc/train.parquet
MODEL_NAME=llama3.2-3b-sft-500
SAVE_DIR=results/medcalc/analysis


CUDA_VISIBLE_DEVICES=7 python src/eval/medcalc/analysis_pass@k.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --category_sample_size 500