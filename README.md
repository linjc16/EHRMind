# EHRMind: Training LLMs for EHR-Based Reasoning Tasks via Reinforcement Learning
Our repo is based on [VeRL](https://github.com/volcengine/verl) framework.


## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Get started

**Data Preparation**
```
conda activate zero
python src/dataset/medcalc/rl/data_process.py
```

### Run Training
```
conda activate zero
```


**3B+ model**
```
export N_GPUS=2
export BASE_MODEL=meta-llama/Meta-Llama-3-8B
export DATA_DIR=data/local_index_search/medcalc
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=medcalc-llama-3-3b-inst-grpo
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="[Your_key]"

export CUDA_VISIBLE_DEVICES=0,1

bash scripts/train/train_medcalc_3b.sh
```