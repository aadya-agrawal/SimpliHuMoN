import numpy as np

if not hasattr(np, "int"):
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float64  
    np.complex = np.complex128
    np.object = np.object_
    np.unicode = np.str_
    np.str = np.str_

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
import inspect as _inspect

if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec

from tqdm import tqdm
import torch
import smplx


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    # Window sizes (in frames)
    past_frames: int = 25  
    future_frames: int = 25  
    
    # Data paths
    data_dir: Path = Path("data/poses")
    output_dir: Path = Path("data/processed")
    
    # Temporal split ratios
    train_ratio: float = 0.8
    split_seed: int = 42
    
    # Missing data handling
    max_missing_ratio: float = 0.5  # Skip windows with >50% missing data
    forward_fill_threshold: int = 5  # Forward fill gaps <= 5 frames
    
    # SMPL model path
    smpl_model_path: str = "data/body_models" 
    smpl_gender: str = "neutral"  
    
    # SMPL dimensions
    body_pose_dim: int = 69  # 23 joints × 3 
    transl_dim: int = 3
    global_orient_dim: int = 3  # Root orientation
    num_joints: int = 24  # (including root)
    
    field_center: Optional[np.ndarray] = None  # [x, y, z] in meters
    normalize_to_field: bool = False
    
    compute_velocities: bool = True  
    include_global_orient: bool = True  


class WorldPoseProcessor:
    """Process WorldPose dataset for motion prediction."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalization_stats = {}
        
        # Initialize SMPL model
        print(f"Loading SMPL model ({config.smpl_gender})...")
        try:
            # Move to GPU if available first
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use smplx.create function
            self.smpl_model = smplx.create(
                model_path=config.smpl_model_path,
                model_type='smpl',
                gender=config.smpl_gender,
                num_betas=10,
            ).eval().to(self.device)
            
            print(f"SMPL model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading SMPL model: {e}")
            print(f"Please ensure SMPL models are in: {config.smpl_model_path}")
            print(f"Expected structure: {config.smpl_model_path}/smpl/SMPL_{config.smpl_gender.upper()}.pkl")
            raise
        
    def load_sequence(self, npz_path: Path) -> Dict[str, np.ndarray]:
        """Load SMPL parameters from NPZ file."""
        data = dict(np.load(npz_path))
        
        N, T = data['global_orient'].shape[:2]
        print(f"\nLoaded: {npz_path.name}")
        print(f"  Shape: {N} people, {T} frames ({T/25:.1f}s)")
        
        return data
    
    def handle_missing_data(self, data: np.ndarray, threshold: int = 5) -> np.ndarray:
        """
        Handle missing data (NaN values) through forward filling.
        
        Args:
            data: Array with potential NaN values (N, T, ...)
            threshold: Maximum gap size to fill
        
        Returns:
            Processed array with filled gaps
        """
        if not np.isnan(data).any():
            return data
        
        result = data.copy()
        N, T = result.shape[:2]
        
        # Flatten additional dimensions for processing
        original_shape = result.shape
        result_flat = result.reshape(N, T, -1)
        
        # Process each person and feature
        for person_idx in range(N):
            for feat_idx in range(result_flat.shape[2]):
                series = result_flat[person_idx, :, feat_idx]
                
                # Find NaN locations
                nan_mask = np.isnan(series)
                if not nan_mask.any():
                    continue
                
                # Forward fill small gaps
                valid_indices = np.where(~nan_mask)[0]
                if len(valid_indices) == 0:
                    continue
                
                for i in range(len(valid_indices) - 1):
                    gap_start = valid_indices[i] + 1
                    gap_end = valid_indices[i + 1]
                    gap_size = gap_end - gap_start
                    
                    if gap_size <= threshold and gap_size > 0:
                        # Forward fill
                        series[gap_start:gap_end] = series[valid_indices[i]]
                
                result_flat[person_idx, :, feat_idx] = series
        
        # Reshape back
        result = result_flat.reshape(original_shape)
        return result
    
    def compute_field_center(self, all_joints: List[np.ndarray]) -> np.ndarray:
        """
        Estimate field center from player root joint positions across all sequences.
        
        Args:
            all_joints: List of joint arrays (N, T, 24, 3) from different sequences
        
        Returns:
            field_center: [x, y, z] estimated field center
        """
        all_root_positions = []
        
        for joints in all_joints:
            # Get root joint positions (pelvis = joint 0)
            root_positions = joints[:, :, 0, :]  # (N, T, 3)
            
            # Get valid (non-NaN) positions
            valid_mask = ~np.isnan(root_positions).any(axis=-1)
            valid_positions = root_positions[valid_mask]
            if len(valid_positions) > 0:
                all_root_positions.append(valid_positions)
        
        if len(all_root_positions) == 0:
            print("Warning: No valid positions found. Using origin as field center.")
            return np.array([0., 0., 0.])
        
        all_root_positions = np.concatenate(all_root_positions, axis=0)
        
        # Use median as robust estimate of field center (X, Y only)
        # Z should be close to ground level
        field_center = np.array([
            np.median(all_root_positions[:, 0]),  # X
            np.median(all_root_positions[:, 1]),  # Y
            np.median(all_root_positions[:, 2])   # Z (approximate ground level)
        ])
        
        print(f"\nEstimated field center: X={field_center[0]:.2f}m, "
              f"Y={field_center[1]:.2f}m, Z={field_center[2]:.2f}m")
        
        return field_center
    
    def normalize_translation(self, transl: np.ndarray) -> np.ndarray:
        """
        Normalize translations relative to field center.
        
        Args:
            transl: (N, T, 3) translation positions
        
        Returns:
            normalized_transl: (N, T, 3) normalized positions
        """
        if self.config.field_center is None:
            return transl
        
        normalized = transl.copy()
        normalized = normalized - self.config.field_center.reshape(1, 1, 3)
        
        return normalized
    
    def compute_velocities(self, data: np.ndarray, fps: float = 25.0) -> np.ndarray:
        """
        Compute velocities using central differences.
        
        Args:
            data: (N, T, D) positions or rotations
            fps: Frames per second
        
        Returns:
            velocities: (N, T, D) velocities
        """
        # Central difference for interior points
        velocities = np.zeros_like(data)
        velocities[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2.0 / fps)
        
        # Forward/backward difference for endpoints
        velocities[:, 0] = (data[:, 1] - data[:, 0]) / (1.0 / fps)
        velocities[:, -1] = (data[:, -1] - data[:, -2]) / (1.0 / fps)
        
        return velocities
    
    def smpl_forward_kinematics(
        self,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        betas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply SMPL forward kinematics to convert rotations to 3D joint positions.
        
        Args:
            body_pose: (N, T, 69) body joint rotations in axis-angle
            global_orient: (N, T, 3) root orientation in axis-angle
            transl: (N, T, 3) root translation
            betas: (N, 10) or (N, T, 10) shape parameters (optional)
        
        Returns:
            joints: (N, T, 24, 3) 3D joint positions in world space
        """
        N, T = body_pose.shape[:2]
        
        # Prepare output
        joints_all = np.zeros((N, T, self.config.num_joints, 3), dtype=np.float32)
        
        with torch.no_grad():
            # Process each person separately (different betas)
            for person_idx in range(N):
                # Handle betas
                if betas is not None:
                    if betas.ndim == 2:  # (N, 10)
                        person_betas = betas[person_idx:person_idx+1].astype(np.float32)  # (1, 10)
                        person_betas = torch.tensor(person_betas, dtype=torch.float32, device=self.device)
                        person_betas = person_betas.expand(T, -1)  # (T, 10)
                    else:  # (N, T, 10)
                        person_betas = betas[person_idx].astype(np.float32)  # (T, 10)
                        person_betas = torch.tensor(person_betas, dtype=torch.float32, device=self.device)
                else:
                    person_betas = None
                
                # Get person's data
                person_body_pose = torch.tensor(body_pose[person_idx], dtype=torch.float32, device=self.device)  # (T, 69)
                person_global_orient = torch.tensor(global_orient[person_idx], dtype=torch.float32, device=self.device)  # (T, 3)
                person_transl = torch.tensor(transl[person_idx], dtype=torch.float32, device=self.device)  # (T, 3)
                
                # Process in batches to avoid OOM
                batch_size = 64
                person_joints = []
                
                for start_idx in range(0, T, batch_size):
                    end_idx = min(start_idx + batch_size, T)
                    
                    batch_body_pose = person_body_pose[start_idx:end_idx]
                    batch_global_orient = person_global_orient[start_idx:end_idx]
                    batch_transl = person_transl[start_idx:end_idx]
                    batch_betas = person_betas[start_idx:end_idx] if person_betas is not None else None
                    
                    # Apply SMPL
                    smpl_output = self.smpl_model(
                        body_pose=batch_body_pose,
                        global_orient=batch_global_orient,
                        transl=batch_transl,
                        betas=batch_betas,
                    )
                    
                    # Get joints (B, 24, 3)
                    batch_joints = smpl_output.joints[:, :self.config.num_joints, :].cpu().numpy()
                    person_joints.append(batch_joints)
                
                # Concatenate batches
                person_joints = np.concatenate(person_joints, axis=0)  # (T, 24, 3)
                joints_all[person_idx] = person_joints
        
        return joints_all
    
    def extract_pose_features(self, smpl_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract motion features from SMPL data using forward kinematics.
        
        Args:
            smpl_data: Dictionary with 'transl', 'body_pose', 'global_orient', 'betas'
        
        Returns:
            Dictionary with extracted features including 3D joint positions
        """
        features = {}
        
        # Extract SMPL parameters
        transl = smpl_data['transl']  # (N, T, 3)
        body_pose = smpl_data['body_pose']  # (N, T, 69)
        global_orient = smpl_data['global_orient']  # (N, T, 3)
        betas = smpl_data.get('betas', None)  # (N, 10) or (N, T, 10) or None
        
        # Handle missing data in SMPL parameters first
        transl = self.handle_missing_data(transl, self.config.forward_fill_threshold)
        body_pose = self.handle_missing_data(body_pose, self.config.forward_fill_threshold)
        global_orient = self.handle_missing_data(global_orient, self.config.forward_fill_threshold)
        
        # Apply SMPL forward kinematics to get 3D joint positions
        print("  Applying SMPL forward kinematics...")
        joints_3d = self.smpl_forward_kinematics(
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            betas=betas
        )  # (N, T, 24, 3)
        
        features['joints_3d'] = joints_3d
        features['root_transl'] = joints_3d[:, :, 0, :]  # (N, T, 3)
        
        # Normalize joint positions if requested
        if self.config.normalize_to_field and self.config.field_center is not None:
            # Normalize all joints relative to field center
            joints_3d_normalized = joints_3d - self.config.field_center.reshape(1, 1, 1, 3)
            features['joints_3d'] = joints_3d_normalized
            features['root_transl'] = joints_3d_normalized[:, :, 0, :]
        
        # Compute velocities if requested
        if self.config.compute_velocities:
            # Compute joint velocities (N, T, 24, 3)
            features['joints_3d_vel'] = self.compute_velocities(joints_3d)
            features['root_transl_vel'] = self.compute_velocities(features['root_transl'])
        
        return features
    
    def create_sliding_windows(
        self, 
        features: Dict[str, np.ndarray],
        sequence_name: str = ""
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create sliding windows for motion prediction.
        
        Args:
            features: Dictionary with pose features (now includes joints_3d)
            sequence_name: Name of the sequence
        
        Returns:
            List of window dictionaries containing past and future data
        """
        # Use joints_3d to determine dimensions
        joints_3d = features['joints_3d']  # (N, T, 24, 3)
        N, T = joints_3d.shape[:2]
        window_size = self.config.past_frames + self.config.future_frames
        stride = self.config.past_frames // 2  # 50% overlap
        
        windows = []
        
        for start_idx in range(0, T - window_size + 1, stride):
            end_idx = start_idx + window_size
            past_end = start_idx + self.config.past_frames
            
            # Check for excessive missing data in joints (key feature)
            window_joints = joints_3d[:, start_idx:end_idx]
            missing_ratio = np.isnan(window_joints).any(axis=-1).any(axis=-1).mean()
            if missing_ratio > self.config.max_missing_ratio:
                continue
            
            # Create valid player mask (players with at least some valid data)
            valid_mask = ~np.isnan(window_joints).all(axis=(1, 2, 3))
            
            if valid_mask.sum() == 0:
                continue
            
            # Create window data
            window_data = {
                'sequence_name': sequence_name,
                'start_frame': start_idx,
                'valid_mask': valid_mask,  # (N,)
            }
            
            # Split all features into past and future
            for key, data in features.items():
                window_slice = data[:, start_idx:end_idx]
                window_data[f'past_{key}'] = window_slice[:, :self.config.past_frames]
                window_data[f'future_{key}'] = window_slice[:, self.config.past_frames:]
            
            windows.append(window_data)
        
        return windows
    
    def split_train_test_by_sequence(
        self, 
        windows: List[Dict[str, np.ndarray]]
    ) -> Tuple[List[Dict], List[Dict], List[str], List[str]]:
        """
        Split windows into train and validation sets by full sequence holdout.
        
        Args:
            windows: List of all windows
        
        Returns:
            train_windows, test_windows, train_sequence_ids, test_sequence_ids
        """
        # Group windows by sequence
        sequence_windows = {}
        for window in windows:
            seq_name = window['sequence_name']
            if seq_name not in sequence_windows:
                sequence_windows[seq_name] = []
            sequence_windows[seq_name].append(window)
        
        seq_names = sorted(sequence_windows.keys())
        num_sequences = len(seq_names)
        test_ratio = 1.0 - self.config.train_ratio

        if num_sequences <= 1:
            num_test_sequences = 0
        else:
            num_test_sequences = max(1, int(round(num_sequences * test_ratio)))

        rng = np.random.default_rng(self.config.split_seed)
        shuffled_seq_names = seq_names.copy()
        rng.shuffle(shuffled_seq_names)

        test_sequence_ids = sorted(shuffled_seq_names[:num_test_sequences]) if num_test_sequences > 0 else []
        train_sequence_ids = sorted(shuffled_seq_names[num_test_sequences:])

        test_sequences = set(test_sequence_ids)

        train_windows = []
        test_windows = []
        
        for seq_name, seq_windows in sequence_windows.items():
            seq_windows.sort(key=lambda x: x['start_frame'])

            if seq_name in test_sequences:
                test_windows.extend(seq_windows)
            else:
                train_windows.extend(seq_windows)
        
        print(
            f"\nSequence-level split (train/test): {len(train_windows)}/{len(test_windows)} windows "
            f"across {len(train_sequence_ids)}/{len(test_sequence_ids)} sequences "
            f"(seed={self.config.split_seed})"
        )
        
        return train_windows, test_windows, train_sequence_ids, test_sequence_ids
    
    def compute_normalization_stats(self, train_windows: List[Dict]) -> Dict:
        """
        Compute normalization statistics from training data.
        
        Args:
            train_windows: List of training windows
        
        Returns:
            Dictionary with mean and std for each feature
        """
        # Collect data for each feature type
        feature_data = {
            'joints_3d': [],
            'root_transl': [],
            'joints_3d_vel': [],
            'root_transl_vel': [],
        }
        
        for window in train_windows:
            valid_mask = window['valid_mask']
            
            for feature_type in feature_data.keys():
                # Check for past and future versions
                past_key = f'past_{feature_type}'
                future_key = f'future_{feature_type}'
                
                if past_key in window:
                    past_data = window[past_key][valid_mask]
                    # Handle potential NaNs
                    if past_data.ndim == 4:  # (N_valid, T, J, 3)
                        past_data = past_data.reshape(-1, past_data.shape[-2], past_data.shape[-1])
                        valid_past = past_data[~np.isnan(past_data).any(axis=(1, 2))]
                        if len(valid_past) > 0:
                            # Flatten joints dimension: (N_valid, J, 3) -> (N_valid, J*3)
                            feature_data[feature_type].append(valid_past.reshape(-1, valid_past.shape[-2] * valid_past.shape[-1]))
                    elif past_data.ndim == 3:  # (N_valid, T, 3)
                        past_data = past_data.reshape(-1, past_data.shape[-1])
                        valid_past = past_data[~np.isnan(past_data).any(axis=-1)]
                        if len(valid_past) > 0:
                            feature_data[feature_type].append(valid_past)
                
                if future_key in window:
                    future_data = window[future_key][valid_mask]
                    # Handle potential NaNs
                    if future_data.ndim == 4:  # (N_valid, T, J, 3)
                        future_data = future_data.reshape(-1, future_data.shape[-2], future_data.shape[-1])
                        valid_future = future_data[~np.isnan(future_data).any(axis=(1, 2))]
                        if len(valid_future) > 0:
                            # Flatten joints dimension: (N_valid, J, 3) -> (N_valid, J*3)
                            feature_data[feature_type].append(valid_future.reshape(-1, valid_future.shape[-2] * valid_future.shape[-1]))
                    elif future_data.ndim == 3:  # (N_valid, T, 3)
                        future_data = future_data.reshape(-1, future_data.shape[-1])
                        valid_future = future_data[~np.isnan(future_data).any(axis=-1)]
                        if len(valid_future) > 0:
                            feature_data[feature_type].append(valid_future)
        
        # Compute stats
        stats = {}
        for feature_type, data_list in feature_data.items():
            if len(data_list) > 0:
                all_data = np.concatenate(data_list, axis=0)
                stats[f'{feature_type}_mean'] = np.mean(all_data, axis=0).tolist()
                stats[f'{feature_type}_std'] = (np.std(all_data, axis=0) + 1e-6).tolist()
        
        print(f"\nNormalization stats computed for {len(stats)//2} feature types")
        
        return stats
    
    def process_all_sequences(self, sequence_paths: List[Path]) -> None:
        """
        Process all sequences and save processed data.
        
        Args:
            sequence_paths: List of paths to NPZ files
        """
        print("="*80)
        print("WORLDPOSE MOTION PREDICTION DATA PROCESSING")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Past frames: {self.config.past_frames} ({self.config.past_frames/25:.2f}s)")
        print(f"  Future frames: {self.config.future_frames} ({self.config.future_frames/25:.2f}s)")
        print(f"  Train ratio: {self.config.train_ratio}")
        print(f"  Split seed: {self.config.split_seed}")
        print(f"  Sequences to process: {len(sequence_paths)}")
        print(f"  SMPL joints: {self.config.num_joints}")
        print(f"  Compute velocities: {self.config.compute_velocities}")
        
        # Step 1: Estimate field center if needed
        if self.config.normalize_to_field and self.config.field_center is None:
            print("\n[1/5] Estimating field center from joint positions...")
            all_joints = []
            for path in tqdm(sequence_paths, desc="Sampling sequences"):  # Sample first 10
                data = self.load_sequence(path)
                # Compute joints for this sequence
                joints = self.smpl_forward_kinematics(
                    body_pose=data['body_pose'],
                    global_orient=data['global_orient'],
                    transl=data['transl'],
                    betas=data.get('betas', None)
                )
                all_joints.append(joints)
            
            self.config.field_center = self.compute_field_center(all_joints)
        
        # Step 2: Extract features and create windows
        print("\n[2/5] Extracting pose features and creating windows...")
        all_windows = []
        
        for seq_path in tqdm(sequence_paths, desc="Processing sequences"):
            # Load sequence
            smpl_data = self.load_sequence(seq_path)
            
            # Extract features
            features = self.extract_pose_features(smpl_data)
            
            # Create windows
            windows = self.create_sliding_windows(
                features,
                sequence_name=seq_path.stem
            )
            
            all_windows.extend(windows)
        
        print(f"\nTotal windows created: {len(all_windows)}")
        
        # Step 3: Train/test split
        print("\n[3/5] Splitting into train/test (random full-sequence holdout)...")
        train_windows, test_windows, train_sequence_ids, test_sequence_ids = self.split_train_test_by_sequence(all_windows)
        
        # Step 4: Compute normalization stats
        print("\n[4/5] Computing normalization statistics...")
        self.normalization_stats = self.compute_normalization_stats(train_windows)
        
        # Step 5: Save processed data
        print("\n[5/5] Saving processed data...")
        
        # Save train data
        train_path = self.config.output_dir / "train_data.npz"
        np.savez_compressed(train_path, windows=train_windows)
        print(f"  Saved: {train_path}")
        
        # Save test data
        test_path = self.config.output_dir / "test_data.npz"
        np.savez_compressed(test_path, windows=test_windows)
        print(f"  Saved: {test_path}")
        
        # Save metadata
        metadata = {
            'config': {
                'past_frames': self.config.past_frames,
                'future_frames': self.config.future_frames,
                'train_ratio': self.config.train_ratio,
                'split_seed': self.config.split_seed,
                'num_joints': self.config.num_joints,
                'compute_velocities': self.config.compute_velocities,
                'smpl_gender': self.config.smpl_gender,
            },
            'field_center': self.config.field_center.tolist() if self.config.field_center is not None else None,
            'normalization_stats': self.normalization_stats,
            'num_train_windows': len(train_windows),
            'num_test_windows': len(test_windows),
            'train_sequence_ids': train_sequence_ids,
            'test_sequence_ids': test_sequence_ids,
            'sequences_processed': [p.stem for p in sequence_paths],
        }
        
        metadata_path = self.config.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {metadata_path}")
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)
        print(f"\nOutput directory: {self.config.output_dir}")
        print(f"  - train_data.npz: {len(train_windows)} windows")
        print(f"  - test_data.npz: {len(test_windows)} windows")
        print(f"  - metadata.json: Configuration and statistics")
        
        # Print feature summary
        sample_window = train_windows[0]
        print(f"\nFeature dimensions per window:")
        for key in sample_window.keys():
            if key.startswith('past_') or key.startswith('future_'):
                data = sample_window[key]
                if isinstance(data, np.ndarray):
                    print(f"  {key}: {data.shape}")


def main():
    """Main processing script."""
    # Use the configuration from ProcessingConfig dataclass
    config = ProcessingConfig()
    
    # Find all NPZ files
    sequence_paths = sorted(config.data_dir.glob("*.npz"))
    
    if len(sequence_paths) == 0:
        print(f"Error: No NPZ files found in {config.data_dir}")
        return
    
    print(f"Found {len(sequence_paths)} sequences")
    
    # Process
    processor = WorldPoseProcessor(config)
    processor.process_all_sequences(sequence_paths)


if __name__ == "__main__":
    main()
