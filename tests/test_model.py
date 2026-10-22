import torch
from models.attention_unet import AttentionUNet

def test_attention_unet_forward():
    model = AttentionUNet(in_channels=8, out_channels=1)
    model.eval()
    x = torch.randn(2, 8, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {out.shape}"
    assert not torch.isnan(out).any(), "Model output contains NaNs"
    print("test_attention_unet_forward passed successfully.")

if __name__ == "__main__":
    test_attention_unet_forward()
