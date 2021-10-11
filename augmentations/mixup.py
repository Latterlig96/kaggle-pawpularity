import torch
import numpy as np

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    if not alpha > 0:
        raise ValueError("alpha should be larger than 0")
    if not x.size(0) > 1:
        raise ValueError("Mixup cannot be applied to a single instance.")

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    
    return mixed_x, target_a, target_b, lam
