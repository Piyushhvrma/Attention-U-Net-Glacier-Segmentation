import os
import numpy as np
import torch
from src.dataset import GlacierDataset

def test_glacier_dataset(tmp_path):
    # Create dummy numpy files
    os.makedirs(tmp_path, exist_ok=True)
    patch = np.random.rand(8, 256, 256).astype(np.float32)
    mask = np.random.randint(0, 2, (256, 256)).astype(np.float32)
    
    np.save(os.path.join(tmp_path, "sample_000_patch.npy"), patch)
    np.save(os.path.join(tmp_path, "sample_000_mask.npy"), mask)
    
    dataset = GlacierDataset(tmp_path)
    assert len(dataset) == 1
    
    img, msk = dataset[0]
    assert img.shape == (8, 256, 256)
    assert msk.shape == (1, 256, 256)
    print("test_glacier_dataset passed successfully.")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_glacier_dataset(td)
