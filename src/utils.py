import os
import random
import numpy as np
import torch
import yaml

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
