MODEL_PATH=checkpoints/matching-sft/checkpoint-180
DATA_PATH=data/local_index_search/matching/train.parquet
MODEL_NAME=llama3.2-3b-sft-100
SAVE_DIR=results/matching/analysis


CUDA_VISIBLE_DEVICES=6 python src/eval/matching/analysis/analysis_pass@k.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --category_sample_size 100