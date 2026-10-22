import torch

def compute_metrics(logits, targets, threshold=0.5):
    """Computes Dice, IoU, Precision, and Recall for binary segmentation."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum().item()
    
    eps = 1e-8
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
