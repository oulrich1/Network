# Documentation

This directory contains comprehensive documentation for the Neural Network library and experiments.

## Directory Structure

```
docs/
├── README.md                          # This file
└── experiments/
    └── mnist_experiments.md           # MNIST experiment documentation
```

## Documentation Files

### Experiments

- **[mnist_experiments.md](experiments/mnist_experiments.md)**
  - Comprehensive documentation of MNIST digit classification experiments
  - Includes architecture details, hyperparameters, and results
  - Documents optimizer, loss function, and activation function choices
  - Contains reproducibility instructions and code examples

## Quick Links

### Experiment Documentation

1. **MNIST Experiments** - [`experiments/mnist_experiments.md`](experiments/mnist_experiments.md)
   - Full dataset training (60,000 samples)
   - Mini test suite (100 samples)
   - Baseline synthetic data tests
   - Performance metrics and results

### Related Files

- **MNIST Loader:** [`../mnist_loader.h`](../mnist_loader.h)
- **MNIST Training Script:** [`../train_mnist.cpp`](../train_mnist.cpp)
- **MNIST Test Suite:** [`../test_mnist_training.cpp`](../test_mnist_training.cpp)
- **Network Implementation:** [`../network.h`](../network.h)

## Adding New Experiment Documentation

When documenting new experiments, please include:

1. **Architecture Details**
   - Layer configuration
   - Number of neurons per layer
   - Activation functions for each layer

2. **Training Configuration**
   - Optimizer type and parameters
   - Loss function
   - Learning rate
   - Batch size
   - Number of epochs

3. **Results**
   - Accuracy metrics
   - Loss curves
   - Training time
   - Convergence behavior

4. **Reproducibility**
   - Environment setup
   - Build commands
   - Data preparation steps
   - Exact hyperparameters

5. **Code Examples**
   - Data loading
   - Network creation
   - Training loop

## Documentation Standards

- Use Markdown format (`.md` files)
- Include code examples with proper syntax highlighting
- Document all hyperparameters
- Provide reproducibility instructions
- Include performance metrics
- Link to relevant source files

## Contributing

When adding new experiments:

1. Create a new file in `docs/experiments/`
2. Use the MNIST experiments doc as a template
3. Update this README with links to your documentation
4. Ensure all code examples are tested and working
5. Include actual results when possible

---

**Last Updated:** 2025-11-17
