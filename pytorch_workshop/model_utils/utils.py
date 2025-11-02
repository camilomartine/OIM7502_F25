"""Utility functions for model evaluation and prediction."""
import torch


def accuracy_fn(y_true, y_pred):
    """Calculate accuracy between truth labels and predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy as a percentage
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def make_predictions(model, X, device='cpu'):
    """Make predictions with a trained model.

    Args:
        model: Trained PyTorch model
        X: Input data
        device: Device to use for inference

    Returns:
        Binary predictions (0 or 1)
    """
    model.eval()
    with torch.inference_mode():
        X = X.to(device)
        logits = model(X).squeeze()
        predictions = torch.round(torch.sigmoid(logits))
    return predictions