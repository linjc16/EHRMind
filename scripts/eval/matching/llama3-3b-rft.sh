MODEL_PATH=checkpoints/Health-R1-matching/matching-llama3.2-3b-inst-ppo-2gpus/actor/global_step_100
DATA_PATH=data/local_index_search/matching/test.parquet
MODEL_NAME=llama3.2-3b-rft
SAVE_DIR=results/matching/output

CUDA_VISIBLE_DEVICES=5 python src/eval/matching/eval_inst.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \
