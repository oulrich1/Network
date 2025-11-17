# PyTorch Toy Example: Linear Regression vs Neural Network

A simple comparison project demonstrating PyTorch implementations of linear regression and neural networks for fitting the continuous function **y = 2x + 1**.

## Overview

This project provides a sanity check and comparison between:
1. **Linear Regression Model** - Direct approach using `y = wx + b`
2. **Neural Network Model** - Using hidden layers with ReLU activations

Both models are trained to learn the same linear function and their performance is compared.

## Purpose

This serves as a PyTorch comparison to the C++ neural network implementation in the parent repository. It demonstrates:
- How PyTorch simplifies neural network implementation
- The difference between classical linear regression and neural network approaches
- Training convergence and loss visualization
- Model evaluation and parameter inspection

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the comparison script:
```bash
cd pytorch_toy_example
python linear_regression_comparison.py
```

Or from the parent directory:
```bash
python pytorch_toy_example/linear_regression_comparison.py
```

## Output

The script will:
1. Generate 100 training samples from y = 2x + 1 (with small noise)
2. Train both models for 1000 epochs
3. Display training progress every 100 epochs
4. Show learned parameters for the linear regression model
5. Evaluate predictions on test points
6. Generate a comparison visualization saved as `comparison_results.png`

### Expected Results

**Linear Regression Model:**
- Should learn weights very close to: w ≈ 2.0, b ≈ 1.0
- Directly represents the function as y = wx + b

**Neural Network Model:**
- Uses hidden layers to approximate the linear function
- May achieve similar or slightly different MSE
- Demonstrates that neural networks can learn linear functions

## Example Output

```
PyTorch Linear Regression vs Neural Network Comparison
======================================================================
Target function: y = 2x + 1

Training Linear Regression Model...
Epoch [100/1000], Loss: 0.0456
Epoch [200/1000], Loss: 0.0412
...

Linear Regression Results:
==================================================
Learned function: y = 1.9876x + 1.0234
True function:    y = 2.0000x + 1.0000
Weight error: 0.012400
Bias error:   0.023400

Neural Network Results:
==================================================
Neural network learned a non-linear approximation

Test predictions:
  x= -2.0: predicted= -2.9856, true= -3.0000, error=0.014400
  x=  0.0: predicted=  1.0123, true=  1.0000, error=0.012300
  x=  2.0: predicted=  5.0089, true=  5.0000, error=0.008900
```

## Comparison to C++ Implementation

This PyTorch example complements the C++ neural network library in the parent directory by:

| Aspect | C++ Implementation | PyTorch Implementation |
|--------|-------------------|----------------------|
| **Language** | C++ with custom matrix ops | Python with PyTorch |
| **Activation** | Sigmoid | ReLU |
| **Optimization** | Custom backprop | SGD optimizer |
| **Use Case** | Low-level understanding | Rapid prototyping |
| **Performance** | High performance, compiled | Fast development, GPU support |

## Files

- `linear_regression_comparison.py` - Main comparison script
- `README.md` - This file
- `requirements.txt` - Python dependencies
- `comparison_results.png` - Generated visualization (after running)

## Learning Points

1. **Linear Regression** is optimal for linear functions - simpler and more interpretable
2. **Neural Networks** can approximate any function but may be overkill for simple linear relationships
3. Both achieve similar results, validating that neural networks can learn linear mappings
4. Training loss visualization helps understand convergence behavior

## Extension Ideas

- Try fitting non-linear functions (quadratic, sine, etc.)
- Experiment with different network architectures
- Add regularization (L1/L2)
- Compare with the C++ implementation's results
- Implement batch training
- Add validation set monitoring
