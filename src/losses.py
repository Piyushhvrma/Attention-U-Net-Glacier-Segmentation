# src/losses.py
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class HybridBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        super(HybridBCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=4/3, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        # weight for false negatives (catches debris-covered ice)
        self.beta = beta
        self.gamma = gamma  # focal parameter
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha *
                                        FP + self.beta * FN + self.smooth)
        return (1 - tversky)**(1 / self.gamma)
