import numpy as np

def compute_spectral_indices(bands):
    """
    Computes NDSI and NDVI from 6 Sentinel-2 spectral bands.
    bands shape: (6, H, W)
      0: B2 (Blue), 1: B3 (Green), 2: B4 (Red), 3: B8 (NIR), 4: B11 (SWIR1), 5: B12 (SWIR2)
    Returns 8-channel array of shape (8, H, W).
    """
    b3_green = bands[1].astype(np.float32)
    b4_red = bands[2].astype(np.float32)
    b8_nir = bands[3].astype(np.float32)
    b11_swir1 = bands[4].astype(np.float32)

    eps = 1e-8
    ndsi = (b3_green - b11_swir1) / (b3_green + b11_swir1 + eps)
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + eps)

    ndsi_norm = (ndsi + 1.0) / 2.0
    ndvi_norm = (ndvi + 1.0) / 2.0

    return np.concatenate([bands.astype(np.float32), 
                           ndsi_norm[np.newaxis, ...], 
                           ndvi_norm[np.newaxis, ...]], axis=0)

def normalize_channels(channels):
    """Min-Max normalizes channels independently to [0, 1]."""
    norm_channels = np.zeros_like(channels, dtype=np.float32)
    for c in range(channels.shape[0]):
        c_min = channels[c].min()
        c_max = channels[c].max()
        if c_max - c_min > 1e-8:
            norm_channels[c] = (channels[c] - c_min) / (c_max - c_min)
        else:
            norm_channels[c] = channels[c]
    return norm_channels
