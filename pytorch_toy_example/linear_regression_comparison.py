#!/usr/bin/env python3
"""
PyTorch Toy Example: Linear Regression vs Neural Network
Compare traditional linear regression with a neural network for fitting y = 2x + 1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def generate_data(n_samples: int = 100, noise: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for y = 2x + 1 with optional noise"""
    x = torch.linspace(-5, 5, n_samples).reshape(-1, 1)
    # True function: y = 2x + 1
    y = 2 * x + 1 + noise * torch.randn(n_samples, 1)
    return x, y


class LinearRegressionModel(nn.Module):
    """Simple linear regression model: y = wx + b"""
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class NeuralNetworkModel(nn.Module):
    """Neural network with hidden layers to learn y = 2x + 1"""
    def __init__(self, hidden_size: int = 16):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epochs: int = 1000, lr: float = 0.01) -> List[float]:
    """Train a model and return loss history"""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return losses


def evaluate_model(model: nn.Module, name: str):
    """Evaluate and print model parameters"""
    print(f"\n{name} Results:")
    print("=" * 50)

    if isinstance(model, LinearRegressionModel):
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()
        print(f"Learned function: y = {weight:.4f}x + {bias:.4f}")
        print(f"True function:    y = 2.0000x + 1.0000")
        print(f"Weight error: {abs(weight - 2.0):.6f}")
        print(f"Bias error:   {abs(bias - 1.0):.6f}")
    else:
        print("Neural network learned a non-linear approximation")
        # Test on specific points
        test_x = torch.tensor([[-2.0], [0.0], [2.0]])
        with torch.no_grad():
            predictions = model(test_x)
        print("\nTest predictions:")
        for x_val, pred in zip(test_x, predictions):
            true_val = 2 * x_val.item() + 1
            print(f"  x={x_val.item():5.1f}: predicted={pred.item():7.4f}, true={true_val:7.4f}, error={abs(pred.item() - true_val):.6f}")


def plot_results(x: torch.Tensor, y: torch.Tensor,
                 lr_model: nn.Module, nn_model: nn.Module,
                 lr_losses: List[float], nn_losses: List[float]):
    """Create visualization comparing both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Data and predictions
    x_numpy = x.numpy()
    y_numpy = y.numpy()

    with torch.no_grad():
        lr_pred = lr_model(x).numpy()
        nn_pred = nn_model(x).numpy()

    # True function
    x_true = np.linspace(-5, 5, 100)
    y_true = 2 * x_true + 1

    ax1.scatter(x_numpy, y_numpy, alpha=0.5, label='Training Data', s=20)
    ax1.plot(x_true, y_true, 'g-', linewidth=2, label='True: y = 2x + 1')
    ax1.plot(x_numpy, lr_pred, 'r--', linewidth=2, label='Linear Regression')
    ax1.plot(x_numpy, nn_pred, 'b-.', linewidth=2, label='Neural Network')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Model Predictions Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training loss
    ax2.plot(lr_losses, 'r-', label='Linear Regression', alpha=0.7)
    ax2.plot(nn_losses, 'b-', label='Neural Network', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pytorch_toy_example/comparison_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to: pytorch_toy_example/comparison_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("PyTorch Linear Regression vs Neural Network Comparison")
    print("=" * 70)
    print("Target function: y = 2x + 1\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate training data
    print("Generating training data...")
    x_train, y_train = generate_data(n_samples=100, noise=0.2)

    # Train Linear Regression Model
    print("\n" + "=" * 70)
    print("Training Linear Regression Model...")
    print("=" * 70)
    lr_model = LinearRegressionModel()
    lr_losses = train_model(lr_model, x_train, y_train, epochs=1000, lr=0.01)

    # Train Neural Network Model
    print("\n" + "=" * 70)
    print("Training Neural Network Model...")
    print("=" * 70)
    nn_model = NeuralNetworkModel(hidden_size=16)
    nn_losses = train_model(nn_model, x_train, y_train, epochs=1000, lr=0.01)

    # Evaluate both models
    evaluate_model(lr_model, "Linear Regression")
    evaluate_model(nn_model, "Neural Network")

    # Calculate final metrics
    print("\n" + "=" * 70)
    print("Final Comparison:")
    print("=" * 70)

    with torch.no_grad():
        lr_pred = lr_model(x_train)
        nn_pred = nn_model(x_train)

        lr_mse = nn.MSELoss()(lr_pred, y_train).item()
        nn_mse = nn.MSELoss()(nn_pred, y_train).item()

        print(f"Linear Regression - Final MSE: {lr_mse:.6f}")
        print(f"Neural Network    - Final MSE: {nn_mse:.6f}")

        if lr_mse < nn_mse:
            print("\n✓ Linear Regression achieved lower error (as expected for linear function)")
        else:
            print("\n✓ Neural Network achieved comparable or better error")

    # Generate visualization
    print("\nGenerating comparison plots...")
    plot_results(x_train, y_train, lr_model, nn_model, lr_losses, nn_losses)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. Linear regression directly models y = wx + b, perfect for linear functions")
    print("2. Neural network uses non-linear activations but can approximate linear functions")
    print("3. For this simple linear problem, linear regression is more efficient")
    print("4. Both models successfully learn the underlying pattern y ≈ 2x + 1")


if __name__ == "__main__":
    main()
