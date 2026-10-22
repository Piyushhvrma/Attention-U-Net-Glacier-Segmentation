import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.attention_unet import AttentionUNet
from src.preprocessing import compute_spectral_indices, normalize_channels

def main():
    parser = argparse.ArgumentParser(description="Full-Scene Inference with Attention U-Net")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Model weights")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionUNet(in_channels=8, out_channels=1).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint {args.checkpoint}")
    else:
        print("Checkpoint not found. Running inference with uninitialized model.")

    model.eval()
    # Generate a synthetic scene for demonstration
    np.random.seed(123)
    H, W = 512, 512
    bands = np.random.uniform(0.05, 0.2, (6, H, W)).astype(np.float32)
    cy, cx = 256, 256
    y, x = np.ogrid[:H, :W]
    glacier = ((y - cy)**2 / 3600.0 + (x - cx)**2 / 6400.0) <= 1.0
    bands[0][glacier] += 0.5
    bands[1][glacier] += 0.55
    bands[2][glacier] += 0.45
    bands[3][glacier] += 0.65

    scene_8ch = compute_spectral_indices(bands)
    scene_norm = normalize_channels(scene_8ch)
    tensor = torch.from_numpy(scene_norm).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask = (probs >= 0.70).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rgb = np.stack([scene_8ch[3], scene_8ch[2], scene_8ch[1]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB Preview")
    axes[0].axis('off')

    axes[1].imshow(probs, cmap='Blues')
    axes[1].set_title("Glacier Probability Map")
    axes[1].axis('off')

    axes[2].imshow(mask, cmap='bone')
    axes[2].set_title("Binary Mask (Thresh=0.70)")
    axes[2].axis('off')

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "prediction_preview.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    np.save(os.path.join(args.output_dir, "predicted_mask.npy"), mask)
    print(f"Saved prediction visualization to {out_path}")

if __name__ == "__main__":
    main()
