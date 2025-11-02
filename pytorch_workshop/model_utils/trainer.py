"""Training utilities for PyTorch models."""
import torch
from .utils import accuracy_fn


def train_model(model, X_train, y_train, X_test, y_test,
                loss_fn, optimizer, epochs=1000, device='cpu',
                print_every=100, seed=42):
    """Train a binary classification model.

    Args:
        model: PyTorch model to train
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        loss_fn: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        print_every: Print metrics every N epochs
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing training history
    """
    # Set random seed
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Track results
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()

        # 1. Forward pass
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Testing phase
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            # 2. Calculate loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # Store results
        results['train_loss'].append(loss.item())
        results['train_acc'].append(acc)
        results['test_loss'].append(test_loss.item())
        results['test_acc'].append(test_acc)

        # Print progress
        if epoch % print_every == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, "
                  f"Accuracy: {acc:.2f}% | "
                  f"Test Loss: {test_loss:.5f}, "
                  f"Test Accuracy: {test_acc:.2f}%")

    return results


def train_step(model, X, y, loss_fn, optimizer, device='cpu'):
    """Perform a single training step.

    Args:
        model: PyTorch model
        X: Input features
        y: Target labels
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to use

    Returns:
        Tuple of (loss, accuracy)
    """
    model.train()
    X, y = X.to(device), y.to(device)

    # Forward pass
    y_logits = model(X).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss and accuracy
    loss = loss_fn(y_logits, y)
    acc = accuracy_fn(y_true=y, y_pred=y_pred)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc


def test_step(model, X, y, loss_fn, device='cpu'):
    """Perform a single test step.

    Args:
        model: PyTorch model
        X: Input features
        y: Target labels
        loss_fn: Loss function
        device: Device to use

    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    X, y = X.to(device), y.to(device)

    with torch.inference_mode():
        # Forward pass
        y_logits = model(X).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # Calculate loss and accuracy
        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y_true=y, y_pred=y_pred)

    return loss.item(), acc