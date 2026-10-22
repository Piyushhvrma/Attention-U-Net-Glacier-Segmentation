import torch
from src.metrics import compute_metrics

def test_metrics_calculation():
    logits = torch.tensor([[[[100.0, -100.0], [100.0, 100.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    
    metrics = compute_metrics(logits, targets, threshold=0.5)
    assert abs(metrics['dice'] - 1.0) < 1e-5, f"Expected Dice ~1.0, got {metrics['dice']}"
    assert abs(metrics['iou'] - 1.0) < 1e-5, f"Expected IoU ~1.0, got {metrics['iou']}"
    print("test_metrics_calculation passed successfully.")

if __name__ == "__main__":
    test_metrics_calculation()
