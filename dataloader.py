from torch_geometric.data import Dataset
import torch
import glob
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import random
import numpy as np
import pickle as pkl

def create_rotation_matrix(angle, axis='z'):
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    if axis == 'z':
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
    elif axis == 'x':
        rot_matrix = torch.tensor([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ], dtype=torch.float32)
    elif axis == 'y':
        rot_matrix = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=torch.float32)
    
    return rot_matrix

def apply_combined_augmentations(data, config):
    N, T, J, D = data.shape
    augmented = data.clone()

    if config.get('rotation', True) and random.choice([0, 1]):
        num_people = config.get('num_people', 3)  
        batch_size = N // num_people
        
        angles = torch.deg2rad(torch.rand(batch_size) * 360)
        
        rot_matrices = torch.stack([
            create_rotation_matrix(angle, axis='z') for angle in angles
        ]) 
        
        reshaped = augmented.reshape(batch_size, num_people, -1, 3) 
        rotated = torch.bmm(reshaped.reshape(batch_size, -1, 3), rot_matrices.transpose(1, 2))  
        augmented = rotated.reshape(N, T, J, 3)
    
    # Random person permutation within each sequence
    if config.get('permute_p', True) and random.choice([0, 1]):
        num_people = config.get('num_people', 3)  
        batch_size = N // num_people
        
        perm = torch.argsort(torch.rand(batch_size, num_people), dim=-1) 
        idx = torch.arange(batch_size).unsqueeze(-1) 
        
        reshaped = augmented.reshape(batch_size, num_people, T, J, 3)  
        permuted = reshaped[idx, perm]  
        augmented = permuted.reshape(N, T, J, 3)
    
    # Random horizontal flip (X-axis)
    if config.get('horizontal_flip', True) and random.choice([0, 1]):
        augmented = augmented * torch.tensor([-1, 1, 1], dtype=torch.float32)
    
    # Random vertical flip (Y-axis)
    if config.get('vertical_flip', True) and random.choice([0, 1]):
        augmented = augmented * torch.tensor([1, -1, 1], dtype=torch.float32)
    
    # Random scaling
    if config.get('scaling', True) and random.choice([0, 1]):
        scale_range = config.get('scale_range', 0.1)  # ±10%
        scale_factor = 1.0 + (torch.rand(1) * 2 - 1) * scale_range
        augmented = augmented * scale_factor

    # Random translation
    if config.get('translation', True) and random.choice([0, 1]):
        trans_range = config.get('translation_range', 0.1)
        translation = (torch.rand(3) * 2 - 1) * trans_range
        augmented = augmented + translation
    return augmented

def eth_ucy_collate(batch):
    input_seqs = []
    output_seqs = []
    peds_nums = []
    only_traj_flags = []
    
    for data in batch:
        input_seqs.append(data['input_seq'])
        output_seqs.append(data['output_seq'])
        peds_nums.append(data['peds_num_per_scene'])
        only_traj_flags.append(data['only_traj'])
    
    input_seq = torch.cat(input_seqs, dim=0)  # [batch_size, time, 2]
    output_seq = torch.cat(output_seqs, dim=0)  # [batch_size, time, 2]
    peds_num_per_scene = torch.cat(peds_nums, dim=0)  # [batch_size]
    only_traj = only_traj_flags[0]  # All should be the same
    
    return {
        'input_seq': input_seq,
        'output_seq': output_seq,
        'peds_num_per_scene': peds_num_per_scene,
        'only_traj': only_traj
    }

class DatasetSetup(Dataset):
    def __init__(self, dataset, mode=0, device='cuda', input_time=25, on_the_fly_augmentations=False):
        self.dataset = dataset
        self.mode = mode  
        self.on_the_fly_augmentations = on_the_fly_augmentations
        self.traj_abs = None

        if dataset == "mocap_umpm":
            self.num_person = 3
            if mode==0: # Train
                self.data = glob.glob('path/to/mocap_UMPM/train/*.pt')
            else:  # Val, Test
                self.data = glob.glob('path/to/mocap_UMPM/val/*.pt')
        elif dataset == "3dpw":
            self.num_person = 2
            if mode == 0:  # Train
                self.data = glob.glob('path/to/3dpw/train/*.pt')
            else:  # Val, Test
                self.data = glob.glob('path/to/3dpw/val/*.pt')
        elif dataset == "h36m":
            self.num_person = 1
            self.h36m_seqs = []
            self.data_idx = []
            self._load_h36m_data()
        else:
            self.num_person = 1  
            self.only_traj = True
            if mode == 0:  # Train
                raw_data = pkl.load(open(f'path/to/eth-ucy/{dataset}_train.pkl', 'rb'))
            else:  # Val, Test
                raw_data = pkl.load(open(f'path/to/eth-ucy/{dataset}_test.pkl', 'rb'))
            self.traj_abs = torch.from_numpy(raw_data[0]).type(torch.float)  
            print(f"{dataset} dataset: Loaded {self.traj_abs.shape[0]} trajectories in mode {mode}")
        self.device = device
        self.dataset = dataset
        self.input_time = input_time
        print(f"Loaded samples for {dataset} mode={mode} on_the_fly={on_the_fly_augmentations})")

        super(DatasetSetup, self).__init__()

    def get(self, idx):
        if self.traj_abs is not None:
            curr_traj = self.traj_abs[idx:idx+1, :, :]  # [1, seq_len, 2]
            
            if self.on_the_fly_augmentations:
                th = random.random() * np.pi
                cos_th, sin_th = np.cos(th), np.sin(th)
                x, y = curr_traj[0, :, 0], curr_traj[0, :, 1]
                curr_traj[0, :, 0] = x * cos_th - y * sin_th
                curr_traj[0, :, 1] = x * sin_th + y * cos_th
            
            data = {
                'input_seq': curr_traj[:, :self.input_time, :],      # [1, 8, 2]
                'output_seq': curr_traj[:, self.input_time:, :],     # [1, 12, 2]
                'peds_num_per_scene': torch.tensor([1]),             # Always 1 pedestrian
                'only_traj': self.only_traj
            }

        elif self.dataset == "h36m":
            seq_idx, start_frame = self.data_idx[idx]
            frame_indexes = np.arange(start_frame, start_frame + self.input_time + 100)
            motion = self.h36m_seqs[seq_idx][frame_indexes]
            
            if self.on_the_fly_augmentations:
                aug_config = {
                    'num_people': self.num_person,
                    'rotation': True,
                    'horizontal_flip': True,
                    'vertical_flip': True,
                }
                
                motion_tensor = torch.from_numpy(motion).float()
                motion_reshaped = motion_tensor.reshape(1, motion.shape[0], -1, 3)
                
                motion_augmented = apply_combined_augmentations(motion_reshaped, aug_config)
                motion = motion_augmented.reshape(motion.shape).numpy()
            
            input_seq = motion[:self.input_time]
            output_seq = motion[self.input_time:]
            
            input_seq = input_seq.reshape(self.input_time, -1)
            output_seq = output_seq.reshape(100, -1)
            
            data = {
                'input_seq': torch.from_numpy(input_seq).float(),
                'output_seq': torch.from_numpy(output_seq).float(),
            }

        else:
            data = torch.load(self.data[idx], map_location='cpu', weights_only=False)
            to_seq = torch.as_tensor(data.body_xyz, dtype=torch.float32)
            
            if self.on_the_fly_augmentations:
                aug_config = {
                    'num_people': self.num_person,
                    'rotation': True,
                    'horizontal_flip': True,
                    'vertical_flip': True,
                    'scaling': True,
                    'scale_range': 0.1,
                    'translation': False,
                    'translation_range': 0,
                    'permute_p': False,
                    'time_warping': False,
                }        
                to_seq = apply_combined_augmentations(to_seq, aug_config)
            
            data['input_seq'] = to_seq[:, :self.input_time, :].reshape(self.num_person, self.input_time, -1)
            data['output_seq'] = to_seq[:, self.input_time:, :].reshape(self.num_person, to_seq.shape[1] - self.input_time, -1)
            data.bos_mask = data.bos_mask.cpu()
        
        return data

    def _load_h36m_data(self):
        h36m_anno_dir = 'data/h3.6m'
        
        if self.mode == 0:  # Train
            data_file = os.path.join(h36m_anno_dir, "data_3d_h36m.npz")
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        else:  # Val/Test
            data_file = os.path.join(h36m_anno_dir, "data_3d_h36m_test.npz")
            subjects = ['S9', 'S11']
            
        data_o = np.load(data_file, allow_pickle=True)
        if 'positions_3d' in data_o:
            positions = data_o['positions_3d'].item()
            is_test_data = False
        else:
            test_data = data_o['data']  
            is_test_data = True

        removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        kept_joints = np.array([x for x in range(32) if x not in removed_joints])
        
        idx = 0
        
        if is_test_data:
            for i in range(len(test_data)):
                pose_data = test_data[i].reshape(125, 48)
                self.h36m_seqs.append(pose_data) 
                
                self.data_idx.append((idx, 0))
                idx += 1
        else:
            for subject in subjects:
                if subject not in positions:
                    continue
                    
                subject_data = positions[subject]
                
                for action_name, action_data in subject_data.items():
                    pose_data = action_data[:, kept_joints, :] 
                    
                    # print(pose_data.shape)
                    pose_data = pose_data[:, 1:] - pose_data[:, 0:1]
                    # print(pose_data_relative.shape)
                    # pose_data = pose_data[:, 1:, :] 
                    
                    N = len(pose_data)
                    if N < 125:  
                        continue
                    
                    pose_data = pose_data.reshape(N, -1)
                    self.h36m_seqs.append(pose_data)
                    
                    valid_frames = np.arange(0, N - 125 + 1, 1)
                    self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                    idx += 1
        print(f"Loaded {len(self.h36m_seqs)} sequences with {len(self.data_idx)} total samples from .npz files")

    def len(self):
        if self.traj_abs is not None:
            return self.traj_abs.shape[0]  
        elif self.dataset == "h36m":
            return len(self.data_idx)
        else:
            return len(self.data)

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name="mocap_umpm", batch_size=64, num_workers=0, 
                 input_time=25, device="cuda", on_the_fly_augmentations=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_time = input_time
        self.device = device
        self.on_the_fly_augmentations = on_the_fly_augmentations

    def setup(self, stage=None):
        self.train_dataset = DatasetSetup(
            self.dataset_name, mode=0, device=self.device, input_time=self.input_time, 
            on_the_fly_augmentations=self.on_the_fly_augmentations,
        )
        self.val_dataset = DatasetSetup(
            self.dataset_name, mode=1, device=self.device, input_time=self.input_time,
            on_the_fly_augmentations=False
        )

    def train_dataloader(self): 
        if self.dataset_name == "eth" or self.dataset_name == "zara1" or self.dataset_name == "zara2" or self.dataset_name == "hotel" or self.dataset_name == "univ" or self.dataset_name == "sdd":
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                num_workers=self.num_workers, pin_memory=True, 
                persistent_workers=True, prefetch_factor=2, collate_fn=eth_ucy_collate
            )
        else:
            return GeoDataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                num_workers=self.num_workers, pin_memory=True, 
                persistent_workers=True, prefetch_factor=2
            )

    def val_dataloader(self):
        if self.dataset_name == "eth" or self.dataset_name == "zara1" or self.dataset_name == "zara2" or self.dataset_name == "hotel" or self.dataset_name == "univ" or self.dataset_name == "sdd":
            return DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False,  
                num_workers=self.num_workers, pin_memory=True, 
                persistent_workers=True, prefetch_factor=2, collate_fn=eth_ucy_collate
            )
        else:
            return GeoDataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False,  
                num_workers=self.num_workers, pin_memory=True, 
                persistent_workers=True, prefetch_factor=2
            )

