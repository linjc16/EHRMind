MODEL_PATH=/shared/eng/jl254/server-05/code/TinyZero/models/llama3
DATA_PATH=data/local_index_search/matching/train.parquet
MODEL_NAME=llama3.2-3b-100
SAVE_DIR=results/matching/analysis


CUDA_VISIBLE_DEVICES=6 python src/eval/matching/analysis/analysis_pass@k.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --category_sample_size 100