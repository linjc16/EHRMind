MODEL_PATH=/shared/eng/jl254/server-05/code/TinyZero/models/llama3
DATA_PATH=data/local_index_search/medcalc/train.parquet
MODEL_NAME=llama3.2-3b
SAVE_DIR=results/medcalc/analysis


CUDA_VISIBLE_DEVICES=5 python src/eval/medcalc/analysis_pass@k.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \