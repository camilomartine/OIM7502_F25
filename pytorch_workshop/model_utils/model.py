"""Model definitions for binary classification."""
import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    """A simple binary classification neural network.

    Architecture:
        - Input layer: 2 features
        - Hidden layer 1: 16 neurons
        - Hidden layer 2: 16 neurons
        - Output layer: 1 neuron (for binary classification)

    Args:
        input_size: Number of input features (default: 2)
        hidden_1: Number of neurons in first hidden layer (default: 16)
        hidden_2: Number of neurons in second hidden layer (default: 16)
        output_size: Number of output neurons (default: 1)

    Example:
        >>> model = BinaryClassifier().to('cuda')
        >>> output = model(input_tensor)
    """

    def __init__(self, input_size=2, hidden_1=16, hidden_2=16,
                 output_size=1):
        """Initialize the BinaryClassifier."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, output_size)
        )

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.network(x)


def setup_training(model, learning_rate=0.1):
    """Setup loss function and optimizer for training.

    Args:
        model: PyTorch model
        learning_rate: Learning rate for the optimizer

    Returns:
        Tuple of (loss_fn, optimizer)
    """
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    return loss_fn, optimizer