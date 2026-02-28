#!/bin/bash

NUM_FUTURE_STEPS=12
NUM_JOINTS=1
BATCH_SIZE=64
INPUT_TIME=8
DECODER_DEPTH=16
EMBED_DIM=48
DATASET_NAME="zara2"
MODEL_TYPE="trajModel"
NUM_MODES=20
NUM_EPOCHS=300
LOG_FILENAME="zara2_traj_d16_emb48_mm20"
OUTPUT_DIR="output_logs"

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_TYPE"
echo "Modes: $NUM_MODES"
echo "Joints: $NUM_JOINTS"
echo "Embed dim: $EMBED_DIM"
echo "Batch size: $BATCH_SIZE"
echo "Input time: $INPUT_TIME"
echo "Future steps: $NUM_FUTURE_STEPS"
echo "Epochs: $NUM_EPOCHS"
echo "Log filename: $LOG_FILENAME"
echo "=========================================="

CMD="nohup python train.py"
CMD="$CMD --num_future_steps $NUM_FUTURE_STEPS"
CMD="$CMD --num_joints $NUM_JOINTS"
CMD="$CMD --embed_dim $EMBED_DIM"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --input_time $INPUT_TIME"
CMD="$CMD --decoder_depth $DECODER_DEPTH"
CMD="$CMD --dataset_name $DATASET_NAME"
CMD="$CMD --model_type $MODEL_TYPE"
CMD="$CMD --num_modes $NUM_MODES"
CMD="$CMD --num_epochs $NUM_EPOCHS"
CMD="$CMD --log_filename $LOG_FILENAME"
CMD="$CMD > $OUTPUT_DIR/$LOG_FILENAME.log 2>&1 &"

echo "Executing: $CMD"
echo "=========================================="

eval $CMD

TRAINING_PID=$!

echo "Training started with PID: $TRAINING_PID"
echo "Log file: $OUTPUT_DIR/$LOG_FILENAME.log"
echo "To monitor training: tail -f $OUTPUT_DIR/$LOG_FILENAME.log"
echo "To stop training: kill $TRAINING_PID"
echo "=========================================="
