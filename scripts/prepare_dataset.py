import os
import numpy as np
from src.preprocessing import compute_spectral_indices

def generate_mock_dataset(output_dir="data/patches", num_samples=10, size=256):
    """Generates synthetic 8-channel patches and masks for local testing without real data."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} synthetic test patches in '{output_dir}'...")
    
    np.random.seed(42)
    for i in range(num_samples):
        # 6 bands
        bands = np.random.uniform(0.08, 0.25, (6, size, size)).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.float32)
        
        # Add a synthetic glacier blob
        cy, cx = np.random.randint(60, size-60), np.random.randint(60, size-60)
        y, x = np.ogrid[:size, :size]
        glacier = ((y - cy)**2 / 400.0 + (x - cx)**2 / 900.0) <= 1.0
        mask[glacier] = 1.0
        
        bands[0][glacier] += 0.5 # Blue
        bands[1][glacier] += 0.55 # Green
        bands[2][glacier] += 0.45 # Red
        bands[3][glacier] += 0.65 # NIR
        
        eight_ch = compute_spectral_indices(bands)
        
        np.save(os.path.join(output_dir, f"patch_{i:03d}_patch.npy"), eight_ch)
        np.save(os.path.join(output_dir, f"patch_{i:03d}_mask.npy"), mask)
        
    print("Synthetic dataset preparation complete!")

if __name__ == "__main__":
    generate_mock_dataset()
