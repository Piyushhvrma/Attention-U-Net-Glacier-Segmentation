import torch
from tqdm import tqdm
from src.metrics import compute_metrics

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for patches, masks in dataloader:
        patches, masks = patches.to(device), masks.to(device)
        
        optimizer.zero_grade() if hasattr(optimizer, 'zero_grade') else optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * patches.size(0)
        metrics = compute_metrics(logits, masks, threshold=0.5)
        running_dice += metrics['dice'] * patches.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    return epoch_loss, epoch_dice

def validate_model(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    with torch.no_grad():
        for patches, masks in dataloader:
            patches, masks = patches.to(device), masks.to(device)
            logits = model(patches)
            loss = criterion(logits, masks)

            running_loss += loss.item() * patches.size(0)
            metrics = compute_metrics(logits, masks, threshold=threshold)
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
            total_tn += metrics['tn']

    eps = 1e-8
    val_loss = running_loss / len(dataloader.dataset)
    dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)

    return val_loss, {"dice": dice, "iou": iou, "precision": precision, "recall": recall}
