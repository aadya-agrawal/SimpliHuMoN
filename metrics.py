import torch

class ADE(torch.nn.Module):
    """Average Displacement Error metric."""
    def __init__(self, frame_idx=-1):
        super().__init__()
        self.frame_idx = frame_idx
        self.reset()
    
    def reset(self):
        self.sum_error = 0.0
        self.num_samples = 0
    
    def update(self, pred, target):
        if pred.dim() == 5:  
            num_modes, batch_size, time_steps = pred.shape[:3]
            
            pred_flat = pred.reshape(num_modes, batch_size, time_steps, -1)
            target_flat = target.reshape(batch_size, time_steps, -1)
            
            dist = torch.norm(pred_flat - target_flat.unsqueeze(0), dim=3)
            dist_mean = dist.mean(dim=2)
            best_mode_errors = dist_mean.min(dim=0)[0]
            
            error = best_mode_errors.sum()
            self.sum_error += error.item()
            self.num_samples += batch_size
            
        else:  
            pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
            target_flat = target.reshape(target.shape[0], target.shape[1], -1)
            
            dist = torch.norm(pred_flat - target_flat, dim=2)
            dist_mean = dist.mean(dim=1)
            
            error = dist_mean.sum()
            self.sum_error += error.item()
            self.num_samples += pred.shape[0]
    
    def compute(self):
        return self.sum_error / max(self.num_samples, 1)


class FDE(torch.nn.Module):
    """Final Displacement Error metric."""
    def __init__(self, frame_idx=-1):
        super().__init__()
        self.frame_idx = frame_idx
        self.reset()
    
    def reset(self):
        self.sum_error = 0.0
        self.num_samples = 0
    
    def update(self, pred, target):
        if pred.dim() == 5:  
            num_modes, batch_size = pred.shape[:2]
            
            final_pred = pred[:, :, self.frame_idx]
            final_target = target[:, self.frame_idx]
            
            final_pred_flat = final_pred.reshape(num_modes, batch_size, -1)
            final_target_flat = final_target.reshape(batch_size, -1)
            
            dist = torch.norm(final_pred_flat - final_target_flat.unsqueeze(0), dim=2)
            best_mode_errors = dist.min(dim=0)[0]
            
            error = best_mode_errors.sum()
            self.sum_error += error.item()
            self.num_samples += batch_size
        else:  
            final_pred = pred[:, self.frame_idx]
            final_target = target[:, self.frame_idx]
            
            final_pred_flat = final_pred.reshape(final_pred.shape[0], -1)
            final_target_flat = final_target.reshape(final_target.shape[0], -1)
            
            dist = torch.norm(final_pred_flat - final_target_flat, dim=1)
            
            error = dist.sum()
            self.sum_error += error.item()
            self.num_samples += pred.shape[0]
    
    def compute(self):
        return self.sum_error / max(self.num_samples, 1)


class APE(torch.nn.Module):
    """Average Position Error metric."""
    def __init__(self, frame_idx=-1):
        super().__init__()
        self.frame_idx = frame_idx
        self.reset()
        self.scale = 1000
    
    def reset(self):
        self.sum_error = 0.0
        self.num_samples = 0
    
    def update(self, pred, target):
        if pred.dim() == 5:  
            batch_size = pred.shape[1]  
            
            pred = pred - pred[:, :, :, 0:1, :]  # [modes, batch, time, joints, 3]
            target = target - target[:, :, 0:1, :]  # [batch, time, joints, 3]
            
            final_pred = pred[:, :, self.frame_idx]  
            final_target = target[:, self.frame_idx] 
            
            dist = torch.norm(final_pred - final_target.unsqueeze(0), dim=-1) 
            dist_mean = dist.mean(dim=-1)  # [modes, batch]
            
            best_mode_errors = dist_mean.min(dim=0)[0]  # [batch]
            
            error = best_mode_errors.sum()
            self.sum_error += error.item()
            self.num_samples += batch_size
            
        else:  
            pred = pred - pred[:, :, 0:1, :]
            target = target - target[:, :, 0:1, :]
            
            final_pred = pred[:, self.frame_idx]
            final_target = target[:, self.frame_idx]
            
            dist = torch.norm(final_pred - final_target, dim=-1)  # [batch, joints]
            dist_mean = dist.mean(dim=-1)  # [batch]
            
            error = dist_mean.sum()
            self.sum_error += error.item()
            self.num_samples += pred.shape[0]
    
    def compute(self):
        return self.sum_error / max(self.num_samples, 1) * self.scale


class JPE(torch.nn.Module):
    """Joint Position Error metric."""
    def __init__(self, frame_idx=-1):
        super().__init__()
        self.frame_idx = frame_idx
        self.reset()
        self.scale = 1000
    
    def reset(self):
        self.sum_error = 0.0
        self.num_samples = 0
    
    def update(self, pred, target):
        if pred.dim() == 5:  
            batch_size = pred.shape[1]
            
            final_pred = pred[:, :, self.frame_idx]
            final_target = target[:, self.frame_idx]
            
            dist = torch.norm(final_pred - final_target.unsqueeze(0), dim=-1)  # [modes, batch, joints]
            best_mode_errors = dist.min(dim=0)[0]  # [batch, joints]
            
            error = best_mode_errors.sum()
            self.sum_error += error.item()
            self.num_samples += batch_size * final_pred.shape[-2] 
        else:  
            final_pred = pred[:, self.frame_idx]
            final_target = target[:, self.frame_idx]
            
            dist = torch.norm(final_pred - final_target, dim=-1)  # [batch, joints]
            
            error = dist.sum()
            self.sum_error += error.item()
            self.num_samples += pred.shape[0] * pred.shape[-2] 
    
    def compute(self):
        return self.sum_error / max(self.num_samples, 1) * self.scale