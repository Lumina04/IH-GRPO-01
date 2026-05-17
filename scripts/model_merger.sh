HF_MODEL_DIR=xxx
CHECKPOINT_DIR=xxx
STEP=xxx
TARGET_DIR=xxx
mkdir -p $TARGET_DIR

python3 ./model_merger.py \
    --backend fsdp \
    --hf_model_path $HF_MODEL_DIR \
    --local_dir $CHECKPOINT_DIR/global_step_$STEP/actor \
    --target_dir $TARGET_DIR \

cp $HF_MODEL_DIR/tokenizer* $TARGET_DIR
cp $HF_MODEL_DIR/merges.txt $TARGET_DIR
cp $HF_MODEL_DIR/vocab.json $TARGET_DIR
echo "mode save successfully!"