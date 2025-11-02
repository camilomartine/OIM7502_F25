# PyTorch Workflow Review 

**Name:** Ernesto Martinez  
**Library:** PyTorch  
**URL:** https://pytorch.org/docs/  
**Slides Deck:** https://bit.ly/4nCs8Hn  
**Description:** Deep learning built for rapid experimentation; and the foundation of many state-of-the-art models.

## Overview

This educational project provides a comprehensive introduction to PyTorch, starting from tensor creation and manipulation, progressing through matrix operations and culminating in building a complete binary classification neural network from scratch. Students will learn the fundamental building blocks—tensors, layers, activations, and loss functions—before combining them into a working model.

This package includes:
- **model_utils/**: Modular code for import to test what is built in the notebook
- **Pytorch_101.ipynb**: Complete educational notebook with all code and explanations
- **Pytorch_101_Tutorial.ipynb**: Interactive notebook with empty code cells to complete during the presentation

## Installation

```bash
pip install torch
```

**Folder Structure:**
```
your_project/
├── model_utils/
│   ├── __init__.py
│   ├── model.py
│   ├── utils.py
│   └── trainer.py
├── Pytorch_101.ipynb
└── Pytorch_101_Tutorial.ipynb
```

## Quick Start

```python
import torch
from model_utils import BinaryClassifier, setup_training, train_model, make_predictions

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BinaryClassifier().to(device)
loss_fn, optimizer = setup_training(model, learning_rate=0.1)

# Train
results = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=1000,
    device=device,
    print_every=100
)

# Predict
predictions = make_predictions(model, X_test, device=device)
```

## Files Reference

### Model
- `BinaryClassifier(input_size=2, hidden_1=16, hidden_2=16, output_size=1)` - Neural network class
- `setup_training(model, learning_rate=0.1)` - Returns (loss_fn, optimizer)

### Training Functions
- `train_model(...)` - Complete training loop, returns metrics dict
- `train_step(model, X, y, loss_fn, optimizer, device)` - Single training step
- `test_step(model, X, y, loss_fn, device)` - Single test step

### Utilities
- `accuracy_fn(y_true, y_pred)` - Calculate accuracy percentage
- `make_predictions(model, X, device)` - Get predictions from trained model

## Custom Training Loop

```python
from model_utils import BinaryClassifier, setup_training, train_step, test_step

model = BinaryClassifier().to(device)
loss_fn, optimizer = setup_training(model, learning_rate=0.1)

for epoch in range(1000):
    train_loss, train_acc = train_step(model, X_train, y_train, 
                                       loss_fn, optimizer, device)
    test_loss, test_acc = test_step(model, X_test, y_test, loss_fn, device)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
```

## Troubleshooting

**Device Error:** Ensure CPU/GPU tensors are on the same device (use `.to(device)`)  
**Matrix Multiplication:** Use appropriate matrix operations (`@` or `torch.matmul()`)  
**Shape Mismatch:** Check tensor dimensions match for operations