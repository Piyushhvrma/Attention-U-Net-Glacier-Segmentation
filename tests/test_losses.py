import torch
from src.losses import HybridBCEDiceLoss, DiceLoss

def test_hybrid_loss():
    criterion = HybridBCEDiceLoss(alpha=0.5)
    logits = torch.randn(2, 1, 256, 256, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    loss = criterion(logits, targets)
    assert not torch.isnan(loss), "Loss is NaN"
    loss.backward()
    assert logits.grad is not None, "Gradients not computed"
    print("test_hybrid_loss passed successfully.")

if __name__ == "__main__":
    test_hybrid_loss()
