#!/usr/bin/env python3
"""
Unit tests for PyTorch linear regression comparison
"""

import torch
import torch.nn as nn
from linear_regression_comparison import (
    generate_data,
    LinearRegressionModel,
    NeuralNetworkModel,
    train_model
)


def test_data_generation():
    """Test that data generation works correctly"""
    print("Testing data generation...")
    x, y = generate_data(n_samples=50, noise=0.0)

    # Check shapes
    assert x.shape == (50, 1), f"Expected x.shape=(50, 1), got {x.shape}"
    assert y.shape == (50, 1), f"Expected y.shape=(50, 1), got {y.shape}"

    # Check that without noise, y = 2x + 1
    expected_y = 2 * x + 1
    assert torch.allclose(y, expected_y, atol=1e-6), "Without noise, y should equal 2x + 1"

    print("✓ Data generation test passed")


def test_linear_regression_model():
    """Test that linear regression model can be instantiated and run"""
    print("Testing linear regression model...")
    model = LinearRegressionModel()

    # Test forward pass
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = model(x)

    assert y.shape == (3, 1), f"Expected output shape (3, 1), got {y.shape}"
    print("✓ Linear regression model test passed")


def test_neural_network_model():
    """Test that neural network model can be instantiated and run"""
    print("Testing neural network model...")
    model = NeuralNetworkModel(hidden_size=8)

    # Test forward pass
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = model(x)

    assert y.shape == (3, 1), f"Expected output shape (3, 1), got {y.shape}"
    print("✓ Neural network model test passed")


def test_linear_regression_training():
    """Test that linear regression can learn y = 2x + 1"""
    print("Testing linear regression training...")

    # Generate perfect data (no noise)
    torch.manual_seed(123)
    x, y = generate_data(n_samples=100, noise=0.0)

    # Train model
    model = LinearRegressionModel()
    losses = train_model(model, x, y, epochs=500, lr=0.01)

    # Check that loss decreased
    assert losses[-1] < losses[0], "Loss should decrease during training"

    # Check learned parameters (should be close to w=2, b=1)
    weight = model.linear.weight.item()
    bias = model.linear.bias.item()

    print(f"  Learned: y = {weight:.4f}x + {bias:.4f}")
    print(f"  Target:  y = 2.0000x + 1.0000")

    # With perfect data and enough training, should be very close
    assert abs(weight - 2.0) < 0.1, f"Weight should be close to 2.0, got {weight}"
    assert abs(bias - 1.0) < 0.1, f"Bias should be close to 1.0, got {bias}"

    print("✓ Linear regression training test passed")


def test_neural_network_training():
    """Test that neural network can learn y = 2x + 1"""
    print("Testing neural network training...")

    # Generate perfect data (no noise)
    torch.manual_seed(456)
    x, y = generate_data(n_samples=100, noise=0.0)

    # Train model
    model = NeuralNetworkModel(hidden_size=16)
    losses = train_model(model, x, y, epochs=500, lr=0.01)

    # Check that loss decreased
    assert losses[-1] < losses[0], "Loss should decrease during training"

    # Check predictions on specific points
    with torch.no_grad():
        test_x = torch.tensor([[0.0], [1.0], [2.0]])
        predictions = model(test_x)
        true_y = 2 * test_x + 1

        # Neural network should approximate the function reasonably well
        mse = nn.MSELoss()(predictions, true_y).item()
        print(f"  MSE on test points: {mse:.6f}")
        assert mse < 0.5, f"MSE should be reasonably low, got {mse}"

    print("✓ Neural network training test passed")


def test_model_comparison():
    """Test that both models can learn the same function"""
    print("Testing model comparison...")

    torch.manual_seed(789)
    x, y = generate_data(n_samples=100, noise=0.1)

    # Train both models
    lr_model = LinearRegressionModel()
    nn_model = NeuralNetworkModel(hidden_size=16)

    lr_losses = train_model(lr_model, x, y, epochs=300, lr=0.01)
    nn_losses = train_model(nn_model, x, y, epochs=300, lr=0.01)

    # Both should achieve reasonable error
    with torch.no_grad():
        lr_pred = lr_model(x)
        nn_pred = nn_model(x)

        lr_mse = nn.MSELoss()(lr_pred, y).item()
        nn_mse = nn.MSELoss()(nn_pred, y).item()

        print(f"  Linear Regression MSE: {lr_mse:.6f}")
        print(f"  Neural Network MSE:    {nn_mse:.6f}")

        # Both should have low error
        assert lr_mse < 1.0, f"Linear regression MSE too high: {lr_mse}"
        assert nn_mse < 1.0, f"Neural network MSE too high: {nn_mse}"

    print("✓ Model comparison test passed")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Running PyTorch Model Tests")
    print("=" * 70)
    print()

    try:
        test_data_generation()
        test_linear_regression_model()
        test_neural_network_model()
        test_linear_regression_training()
        test_neural_network_training()
        test_model_comparison()

        print()
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"Test failed: {e}")
        print("=" * 70)
        return 1

    except Exception as e:
        print()
        print("=" * 70)
        print(f"Unexpected error: {e}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
