import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.attention_unet import AttentionUNet
from src.dataset import GlacierDataset
from src.losses import HybridBCEDiceLoss
from src.engine import validate_model
from src.utils import load_config

def main():
    parser = argparse.ArgumentParser(description="Evaluate Attention U-Net")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/patches", help="Path to dataset directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GlacierDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = AttentionUNet(in_channels=8, out_channels=1).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with random weights.")

    criterion = HybridBCEDiceLoss()
    loss, metrics = validate_model(model, loader, criterion, device, threshold=0.70)

    print("="*50)
    print("         EVALUATION RESULTS (Threshold = 0.70)        ")
    print("="*50)
    print(f"Loss:       {loss:.4f}")
    print(f"Dice Score: {metrics['dice']:.4f}")
    print(f"IoU Score:  {metrics['iou']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
