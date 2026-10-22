import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.preprocessing import compute_spectral_indices, normalize_channels

class GlacierDataset(Dataset):
    """
    PyTorch Dataset for loading 8-channel Sentinel-2 patches and binary glacier masks (.npy format).
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            self.patch_files = []
            self.mask_files = []
        else:
            self.patch_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_patch.npy')])
            self.mask_files = [f.replace('_patch.npy', '_mask.npy') for f in self.patch_files]

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        patch = np.load(self.patch_files[idx])
        mask = np.load(self.mask_files[idx])
        
        if patch.shape[0] == 6:
            patch = compute_spectral_indices(patch)
        patch = normalize_channels(patch)
        
        patch_tensor = torch.from_numpy(patch).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return patch_tensor, mask_tensor
