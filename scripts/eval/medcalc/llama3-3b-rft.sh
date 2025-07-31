MODEL_PATH=checkpoints/Health-R1-medcalc/medcalc-all-regu-llama3.2-3b-inst-ppo-4gpus/actor/global_step_1700
DATA_PATH=data/local_index_search/medcalc/test.parquet
MODEL_NAME=llama3.2-3b-rft
SAVE_DIR=results/medcalc


CUDA_VISIBLE_DEVICES=0 python src/eval/medcalc/eval_inst.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \