import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        total_area = probs_flat.sum() + targets_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (total_area + self.smooth)
        return 1.0 - dice

class HybridBCEDiceLoss(nn.Module):
    """Hybrid loss combining BCEWithLogitsLoss and Soft Dice Loss."""
    def __init__(self, alpha=0.5):
        super(HybridBCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss
