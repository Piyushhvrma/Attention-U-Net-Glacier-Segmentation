import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from models.attention_unet import AttentionUNet
from src.dataset import GlacierDataset
from src.losses import HybridBCEDiceLoss
from src.engine import train_epoch, validate_model
from src.utils import seed_everything, load_config

def main():
    parser = argparse.ArgumentParser(description="Train Attention U-Net for Glacier Segmentation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(42)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data_dir = config["data"]["data_dir"]
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("Data directory empty or missing. Generating synthetic test dataset...")
        from scripts.prepare_dataset import generate_mock_dataset
        generate_mock_dataset(output_dir=data_dir, num_samples=20)

    dataset = GlacierDataset(data_dir)
    val_size = int(len(dataset) * (1.0 - config["data"]["train_val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False)

    model = AttentionUNet(in_channels=config["model"]["in_channels"], out_channels=config["model"]["out_channels"]).to(device)
    criterion = HybridBCEDiceLoss(alpha=config["training"]["alpha_bce_dice"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    best_dice = 0.0

    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    for epoch in range(1, config["training"]["epochs"] + 1):
        t_loss, t_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = validate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{config['training']['epochs']} | Train Loss: {t_loss:.4f} (Dice: {t_dice:.4f}) | Val Loss: {v_loss:.4f} (Dice: {v_metrics['dice']:.4f}, IoU: {v_metrics['iou']:.4f})")

        if v_metrics['dice'] > best_dice:
            best_dice = v_metrics['dice']
            ckpt_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved new best model checkpoint with Dice: {best_dice:.4f}")

    print(f"Training complete! Best Validation Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
