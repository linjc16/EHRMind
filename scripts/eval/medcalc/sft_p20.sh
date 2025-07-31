MODEL_PATH=checkpoints/medcalc-sft-20p/checkpoint-238
DATA_PATH=data/local_index_search/medcalc/test.parquet
MODEL_NAME=medcalc-sft-p20-llama3.2-3b
SAVE_DIR=results/medcalc


CUDA_VISIBLE_DEVICES=0 python src/eval/medcalc/eval_inst.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \