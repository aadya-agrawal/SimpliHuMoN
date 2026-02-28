#!/bin/bash

NUM_FUTURE_STEPS=100
NUM_JOINTS=16
BATCH_SIZE=64
INPUT_TIME=25
DECODER_DEPTH=16
EMBED_DIM=48
DATASET_NAME="h36m"
ON_THE_FLY_AUGMENTATIONS=true
MODEL_TYPE="poseModel"
NUM_MODES=7
NUM_EPOCHS=500
LOG_SUBDIR="empmp"
LOG_FILENAME="h36m_pose_d16_emb48_mm7"
OUTPUT_DIR="output_logs"

echo "=========================================="
echo "Pose Prediction Training Configuration"
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
echo "Augmentations: $ON_THE_FLY_AUGMENTATIONS"
echo "Log filename: $LOG_FILENAME"
echo "=========================================="

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

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
if [ "$ON_THE_FLY_AUGMENTATIONS" = true ]; then
    CMD="$CMD --on_the_fly_augmentations"
fi
CMD="$CMD > $OUTPUT_DIR/$LOG_FILENAME.log 2>&1 &"

echo "Executing: $CMD"
echo "=========================================="

eval $CMD

TRAINING_PID=$!

echo "Pose prediction training started with PID: $TRAINING_PID"
echo "Log file: $OUTPUT_DIR/$LOG_FILENAME.log"
echo "To monitor training: tail -f $OUTPUT_DIR/$LOG_FILENAME.log"
echo "To stop training: kill $TRAINING_PID"
echo "=========================================="