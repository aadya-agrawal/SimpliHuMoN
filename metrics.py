import torch


def _mode_errors(pred, target, frame_idx=None, root_relative=False):
    if root_relative:
        pred = pred - pred[..., 0:1, :]
        target = target - target[..., 0:1, :]
    if frame_idx is not None:
        pred = pred[:, :, frame_idx]
        target = target[:, frame_idx]
    distances = torch.norm(pred - target.unsqueeze(0), dim=-1)
    return distances.reshape(distances.shape[0], distances.shape[1], -1).mean(dim=-1)

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
        errors = _mode_errors(pred, target).min(dim=0).values
        self.sum_error += errors.sum().item()
        self.num_samples += target.shape[0]
    
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
        errors = _mode_errors(pred, target, frame_idx=self.frame_idx).min(dim=0).values
        self.sum_error += errors.sum().item()
        self.num_samples += target.shape[0]
    
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
        errors = _mode_errors(
            pred, target, frame_idx=self.frame_idx, root_relative=True
        ).min(dim=0).values
        self.sum_error += errors.sum().item()
        self.num_samples += target.shape[0]
    
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
        errors = _mode_errors(pred, target, frame_idx=self.frame_idx).min(dim=0).values
        self.sum_error += errors.sum().item()
        self.num_samples += target.shape[0]
    
    def compute(self):
        return self.sum_error / max(self.num_samples, 1) * self.scale