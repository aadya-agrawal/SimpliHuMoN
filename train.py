#!/usr/bin/env python3

import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import math

# import sys
# sys.path.append(YOUR_PATH_HERE)

from metrics import ADE, FDE, APE, JPE
from dataloader import DataModule
from predictor import TrajectoryPredictor, PosePredictor, PoseTrajPredictor

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class MetricsLogger(pl.Callback):
    """Callback to log final metrics to a text file."""
    def __init__(self, output_file="final_metrics.txt"):
        super().__init__()
        self.output_file = output_file
        self.final_metrics = {}
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Store metrics at the end of each validation epoch."""
        metrics = trainer.callback_metrics
        
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                self.final_metrics[key] = value.item()
            else:
                self.final_metrics[key] = value
    
    def on_fit_end(self, trainer, pl_module):
        """Write final metrics to file."""
        os.makedirs(os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else '.', exist_ok=True)
        
        with open(self.output_file, 'w') as f:
            f.write("FINAL TRAINING METRICS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {pl_module.model_type}\n")
            f.write(f"Dataset: {pl_module.dataset_name}\n")
            f.write(f"Epochs: {trainer.current_epoch + 1}\n")
            f.write("=" * 50 + "\n\n")
            
            if pl_module.model_type == "posetrajModel":
                f.write("POSE+TRAJECTORY METRICS (APE/JPE at final time point):\n")
                f.write("-" * 50 + "\n")
                for key, value in self.final_metrics.items():
                    if 'APE' in key or 'JPE' in key:
                        f.write(f"{key}: {value:.6f}\n")
            else:
                f.write("TRAJECTORY/POSE METRICS (ADE/FDE at final time point):\n")
                f.write("-" * 50 + "\n")
                for key, value in self.final_metrics.items():
                    if 'ADE' in key or 'FDE' in key:
                        f.write(f"{key}: {value:.6f}\n")
            
            f.write("\nALL METRICS:\n")
            f.write("-" * 50 + "\n")
            for key, value in sorted(self.final_metrics.items()):
                f.write(f"{key}: {value:.6f}\n")
        
        print(f"Final metrics saved to: {self.output_file}")

class TrainingModule(pl.LightningModule):    
    def __init__(self, 
                 model_type="posetrajModel",
                 dataset_name="mocap_umpm",
                 num_future_steps=50,
                 num_joints=15,
                 input_time=25,
                 embed_dim=48,
                 num_modes=6,
                 decoder_depth=16,
                 pose_mlp_hidden_dim=None,
                 layer_norm_elementwise_affine=False):
        super().__init__()
        
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.num_future_steps = num_future_steps
        self.num_joints = num_joints
        
        if model_type == "poseModel":
            self.net = PosePredictor(
                nblocks=decoder_depth,
                embed_dim=embed_dim,
                num_modes=num_modes,
                num_joints=num_joints,
                input_time=input_time,
                num_future_steps=num_future_steps
            )
        elif model_type == "trajModel":
            self.net = TrajectoryPredictor(
                nblocks=decoder_depth,
                embed_dim=embed_dim,
                num_modes=num_modes,
                input_time=input_time,
                num_future_steps=num_future_steps
            )
        elif model_type == "posetrajModel":
            self.net = PoseTrajPredictor(
                nblocks=decoder_depth,
                embed_dim=embed_dim,
                num_joints=num_joints,
                input_time=input_time,
                num_future_steps=num_future_steps,
                num_modes=num_modes,
                pose_mlp_hidden_dim=pose_mlp_hidden_dim,
                layer_norm_elementwise_affine=layer_norm_elementwise_affine,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.criterion = lambda pred, target: torch.norm(pred - target, dim=-1).mean()
        
        if model_type == "posetrajModel":
            self.val_ape = APE(frame_idx=-1)
            self.val_jpe = JPE(frame_idx=-1)
        else:
            self.val_ade = ADE(frame_idx=-1)
            self.val_fde = FDE(frame_idx=-1)
    
    def forward(self, data):
        return self.net(data)

    def load_release_state_dict(self, state_dict):
        renames = {
            "scale.": "input_proj.",
            "input_emb.": "inp_enc.",
            "fut_scale.": "fut_proj.",
            "output_emb.": "out_enc.",
            "block.": "layers.",
            "out.": "output_proj.",
            "mlp.": "mlp_dec.",
        }
        converted = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in renames.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            converted[new_key] = value
        self.net.load_state_dict(converted, strict=True)

    def _prepare_target(self, target):
        if self.model_type != "trajModel" and target.dim() == 3:
            target = target.reshape(target.shape[0], target.shape[1], -1, 3)
        return target

    @staticmethod
    def _best_mode(pred, target):
        distances = torch.norm(pred - target.unsqueeze(0), dim=-1)
        mode_errors = distances.flatten(start_dim=2).mean(dim=-1)
        best_mode = mode_errors.argmin(dim=0)
        batch_indices = torch.arange(pred.shape[1], device=pred.device)
        return pred[best_mode, batch_indices]
    
    def training_step(self, batch, batch_idx):
        pred = self(batch)
        target = self._prepare_target(batch['output_seq'])
        pred = self._best_mode(pred, target)
        
        loss = self.criterion(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        target = self._prepare_target(batch['output_seq'])
        val_loss = self.criterion(self._best_mode(pred, target), target)
        
        self.log('val_loss', val_loss, prog_bar=True)
    
        if self.model_type == "posetrajModel":
            self.val_ape.update(pred, target)
            self.val_jpe.update(pred, target)
        else:
            self.val_ade.update(pred, target)
            self.val_fde.update(pred, target)    
        return val_loss
    
    def on_validation_epoch_end(self):
        if self.model_type == "posetrajModel":
            ape_val = self.val_ape.compute()
            jpe_val = self.val_jpe.compute()
            self.log("val_APE", ape_val, prog_bar=True)
            self.log("val_JPE", jpe_val, prog_bar=True)
            self.val_ape.reset()
            self.val_jpe.reset()
        else:
            ade_val = self.val_ade.compute()
            fde_val = self.val_fde.compute()
            self.log("val_ADE", ade_val, prog_bar=True)
            self.log("val_FDE", fde_val, prog_bar=True)
            self.val_ade.reset()
            self.val_fde.reset()
    
    def configure_optimizers(self):
        base_lr   = 1e-3 
        min_lr    = 1e-4 
        warmup_steps = 100  
        total_steps = self.trainer.estimated_stepping_batches 

        optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr, betas=(0.95, 0.999), weight_decay=1e-4)

        # warm-up schedule
        def lr_lambda_warmup(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # cosine schedule afterwards
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr / base_lr + (1 - min_lr / base_lr) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda_warmup
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simplified SimpliHuMoN Training')
    
    parser.add_argument('--model_type', type=str, default='posetrajModel', 
                       choices=['poseModel', 'trajModel', 'posetrajModel'],
                       help='Model type to train')
    parser.add_argument('--dataset_name', type=str, default='mocap_umpm',
                       help='Dataset name')
    parser.add_argument('--num_future_steps', type=int, default=50,
                       help='Number of future steps to predict')
    parser.add_argument('--num_joints', type=int, default=15,
                       help='Number of joints')
    parser.add_argument('--input_time', type=int, default=25,
                       help='Input time steps')
    parser.add_argument('--embed_dim', type=int, default=48,
                       help='Embedding dimension')
    parser.add_argument('--num_modes', type=int, default=6,
                       help='Number of modes for multimodal prediction')
    parser.add_argument('--decoder_depth', type=int, default=16,
                       help='Decoder depth')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of devices')
    
    parser.add_argument('--on_the_fly_augmentations', action='store_true',
                       help='Enable on-the-fly augmentations')
    
    parser.add_argument('--log_filename', type=str, default='simplified_training',
                       help='Log filename')
    parser.add_argument('--output_dir', type=str, default='output_logs',
                       help='Output directory')
    parser.add_argument('--checkpoint_path', type=str,
                       help='Checkpoint to load for evaluation')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Run validation from a checkpoint without training')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    if args.evaluate_only and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required with --evaluate_only")

    release_checkpoint = None
    if args.evaluate_only and args.checkpoint_path.endswith(".pt"):
        release_checkpoint = torch.load(
            args.checkpoint_path, map_location="cpu", weights_only=True
        )
        config = release_checkpoint["config"]
        for name in (
            "model_type", "dataset_name", "num_future_steps", "num_joints",
            "input_time", "embed_dim", "num_modes", "decoder_depth",
        ):
            setattr(args, name, config[name])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    datamodule = DataModule(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=4,
        input_time=args.input_time,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        on_the_fly_augmentations=args.on_the_fly_augmentations
    )
    datamodule.setup()
    
    model = TrainingModule(
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        num_future_steps=args.num_future_steps,
        num_joints=args.num_joints,
        input_time=args.input_time,
        embed_dim=args.embed_dim,
        num_modes=args.num_modes,
        decoder_depth=args.decoder_depth,
        pose_mlp_hidden_dim=(
            release_checkpoint["config"].get("pose_mlp_hidden_dim")
            if release_checkpoint else None
        ),
        layer_norm_elementwise_affine=(
            release_checkpoint["config"].get("layer_norm_elementwise_affine", False)
            if release_checkpoint else False
        ),
    )

    if release_checkpoint:
        model.load_release_state_dict(release_checkpoint["state_dict"])
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.log_filename),
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        save_weights_only=False
    )
    
    metrics_logger = MetricsLogger(
        output_file=os.path.join(args.output_dir, f"{args.log_filename}_metrics.txt")
    )
    
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        callbacks=[checkpoint_callback, metrics_logger],
        log_every_n_steps=10,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    
    if args.evaluate_only:
        trainer.validate(
            model,
            dataloaders=datamodule.val_dataloader(),
            ckpt_path=None if release_checkpoint else args.checkpoint_path,
        )
        return

    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
