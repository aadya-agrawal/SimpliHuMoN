# SimpliHuMoN: Simplified Human Motion Prediction

A simplified implementation for human motion prediction supporting three model types: pose prediction, trajectory prediction, and combined pose+trajectory prediction.

<video src="assets/demo.mp4" controls width="100%"></video>

## Overview

SimpliHuMoN provides a clean, streamlined framework for training and evaluating human motion prediction models. The system supports three distinct model architectures:

- **PoseModel**: Predicts human pose sequences (joint positions)
- **TrajModel**: Predicts human trajectory (hip movement)  
- **PoseTrajModel**: Jointly predicts both pose and trajectory

## Features
 
- **Three Model Types**: Specialized architectures for different prediction tasks
- **Multiple Datasets**: Support for H3.6M, ETH-UCY, SDD, Mocap UMPM, AMASS and 3DPW datasets
- **Multimodal Prediction**: Generate multiple plausible future motion sequences
- **Data Augmentation**: On-the-fly augmentations for improved generalization
- **Comprehensive Metrics**: APE/JPE for pose models, ADE/FDE for trajectory models
- **PyTorch Lightning**: Clean training loop with automatic logging and checkpointing

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- PyTorch Lightning 1.8+
- PyTorch Geometric (for some datasets)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning
pip install torch-geometric
pip install numpy
```

### Checkpoints

Pre-trained models will be made available soon.

## Project Structure

```
SimpliHuMoN/
├── README.md                 # This file
├── train.py                  # Main training script
├── predictor.py              # Model architectures
├── dataloader.py             # Data loading and preprocessing
├── metrics.py                # Evaluation metrics
├── pose_training.sh          # Pose model training script
├── traj_training.sh          # Trajectory model training script
└── posetraj_training.sh      # Combined model training script
└── worldpose_processor.py    # Processing script for worldpose data
```

## Model Architectures

### PoseTrajPredictor
- **Purpose**: Jointly predicts human pose and trajectory
- **Input**: Past motion sequences 
- **Output**: Predicted motion sequences 
- **Architecture**: Transformer-based with separate encoders for pose and trajectory
- **Metrics**: APE (Average Position Error), JPE (Joint Position Error)

### PosePredictor  
- **Purpose**: Predicts human pose sequences only
- **Input**: Past pose sequences 
- **Output**: Predicted pose sequences 
- **Architecture**: Transformer-based pose encoder
- **Metrics**: ADE (Average Displacement Error), FDE (Final Displacement Error)

### TrajectoryPredictor
- **Purpose**: Predicts human trajectory
- **Input**: Past trajectory 
- **Output**: Future trajectory
- **Architecture**: Transformer-based trajectory encoder
- **Metrics**: ADE (Average Displacement Error), FDE (Final Displacement Error)

## Datasets
Follow the data loading process below and add the path of the location to ``dataloader.py``

## Usage

### Quick Start

1. **Pose+Trajectory Prediction** (Mocap UMPM dataset):
```bash
./posetraj_training.sh
```

2. **Pose Prediction** (H3.6M dataset):
```bash
./pose_training.sh
```

3. **Trajectory Prediction** (ETH-UCY dataset):
```bash
./traj_training.sh
```

### Manual Training

You can also run training manually with custom parameters:

```bash
python train.py \
    --model_type posetrajModel \
    --dataset_name mocap_umpm \
    --num_future_steps 50 \
    --num_joints 15 \
    --embed_dim 48 \
    --num_modes 6 \
    --decoder_depth 16 \
    --batch_size 64 \
    --num_epochs 300 \
    --on_the_fly_augmentations
```

### Command Line Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model_type` | Model architecture | `posetrajModel` | `poseModel`, `trajModel`, `posetrajModel` |
| `--dataset_name` | Dataset to use | `mocap_umpm` | `h36m`, `mocap_umpm`, `zara2`, etc. |
| `--num_future_steps` | Future prediction length | `50` | Any positive integer |
| `--num_joints` | Number of body joints | `15` | Any positive integer |
| `--embed_dim` | Embedding dimension | `48` | Any positive integer |
| `--num_modes` | Number of prediction modes | `6` | Any positive integer |
| `--decoder_depth` | Transformer decoder depth | `16` | Any positive integer |
| `--batch_size` | Training batch size | `64` | Any positive integer |
| `--num_epochs` | Number of training epochs | `100` | Any positive integer |
| `--on_the_fly_augmentations` | Enable data augmentation | `False` | Flag |

## Datasets

### Supported Datasets

1. **H3.6M/AMASS**: Dataset for pose prediction
   - 25 input frames, 100 output frames
   - Single person sequences
   - Raw files downloading and preprocessing according to [BeLFusion](https://github.com/BarqueroGerman/BeLFusion)

2. **ETH-UCY/SDD**: Pedestrian trajectory datasets
   - 8 input frames, 12 output frames
   - Multiple pedestrians per scene
   - Preprocessed data available at [NMRF](https://github.com/AdaCompNUS/NMRF_TrajectoryPrediction)

3. **Mocap-UMPM/3DPW**: Multi-person interaction datasets
   - 25 input frames, 50 output frames
   - 2-3 people per scene
   - Raw files downloading and preprocessing according to [T2P](https://github.com/jaewoo97/t2p) 

## Hardware Requirements

- **GPU**: Recommended for training (CUDA-compatible)
- **RAM**: 16GB+ recommended
- **Storage**: Varies by dataset (H3.6M ~2GB, Mocap UMPM ~10GB)

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@misc{simplihumon2024,
  title={SimpliHuMoN: Simplified Human Motion Prediction},
  author={Aadya Agrawal},
  year={2024},
  howpublished={GitHub repository},
  url={https://github.com/yourusername/SimpliHuMoN}
}
``` -->
