export SANDBOX_ENDPOINT="http://127.0.0.1:12345/faas/sandbox/"
export TENSORBOARD_DIR="xxx"
MODEL_PATH=xxx
DATA_PATH=xxx
ROLLOUT_SAVE_PATH=xxx
CHECKPOINT_PATH=xxx
LOG_PATH=./logs \
NNODES=1 \
GPUS_PER_NODE=8 \
RESUME=True \
CONFIG_NAME=simpletir_trainer \
bash train.sh \
  --max_response_length 8000 \
  --max_prompt_length 16000 \
  --model_name Qwen3-8B \
  --max_turns 5 \
  --train_batch_size 16 \
  --val_sample_size 50 \
  --n_val 8 \
  --add_c_loss True \
  --train_dataset "simplelr_math_35/train deepscaler/train"

# pip3 install math-verify[antlr4_13_2]
# pip3 install --upgrade omegaconf
# pip3 install word2number
# pip3 install dill
# pip3 install --upgrade packaging
# uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 8
# /home/hadoop-ai-search/.local/bin/hope run ../hope/1_node_jupyter.hope