MODEL_PATH=checkpoints/Health-R1-medcalc/severity-sft-llama3.2-3b-inst-ppo-4gpus/actor/global_step_100
DATA_PATH=data/local_index_search/medcalc_sub/severity/test.parquet
MODEL_NAME=llama3.2-3b-sft-rft
SAVE_DIR=results/medcalc/single_category/severity


CUDA_VISIBLE_DEVICES=5 python src/eval/medcalc/eval_inst.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \