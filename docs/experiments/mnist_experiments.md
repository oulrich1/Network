# MNIST Experiments Documentation

## Overview

This document tracks experiments conducted on the MNIST digit classification dataset using various neural network architectures, optimizers, activation functions, and loss functions.

---

## Dataset Information

**Dataset:** MNIST Handwritten Digit Classification
- **Source:** http://yann.lecun.com/exdb/mnist/
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Input format:** 28×28 grayscale images (784 pixels)
- **Output format:** 10 classes (digits 0-9)
- **Preprocessing:**
  - Pixel normalization to [0.0, 1.0] range
  - One-hot encoding for labels

---

## Experiment 1: Standard MNIST Training (Full Dataset)

### Network Architecture

```
Input Layer:    784 neurons  (28×28 flattened pixels)
                ↓ (ReLU activation)
Hidden Layer 1: 256 neurons
                ↓ (ReLU activation)
Hidden Layer 2: 128 neurons
                ↓ (ReLU activation)
Output Layer:   10 neurons   (Sigmoid activation)
```

**Total Architecture:** 784 → 256 → 128 → 10

### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Cross-Entropy |
| **Learning Rate** | 0.001 |
| **Batch Size** | 32 |
| **Epochs** | 10 (default) |
| **Data Shuffling** | Enabled |

### Activation Functions

- **Hidden Layers:** ReLU (Rectified Linear Unit)
  - Formula: `f(x) = max(0, x)`
  - Gradient: `f'(x) = 1 if x > 0, else 0`

- **Output Layer:** Sigmoid
  - Formula: `f(x) = 1 / (1 + e^(-x))`
  - Gradient: `f'(x) = f(x) * (1 - f(x))`

### Optimizer Details

**Adam (Adaptive Moment Estimation)**
- Combines momentum and RMSprop
- Adaptive learning rates for each parameter
- Default parameters:
  - Beta1: 0.9 (momentum)
  - Beta2: 0.999 (RMSprop)
  - Epsilon: 1e-8

### Loss Function Details

**Cross-Entropy Loss**
- Suitable for multi-class classification
- Formula: `L = -Σ(y_true * log(y_pred))`
- Provides strong gradients for incorrect predictions

### Expected Performance

- **Test Accuracy:** 85-92% after 10-20 epochs
- **Training Time:** ~5-10 minutes on CPU (60,000 samples)
- **Convergence:** Typically within 5-10 epochs

### Implementation Files

- **Trainer:** `train_mnist.cpp` (lines 1-313)
- **Loader:** `mnist_loader.h` (lines 1-290)
- **Network:** `network.h`

---

## Experiment 2: Mini MNIST Test (Small Subset)

### Network Architecture

```
Input Layer:    784 neurons  (28×28 flattened pixels)
                ↓ (ReLU activation)
Hidden Layer:   128 neurons
                ↓ (ReLU activation)
Output Layer:   10 neurons   (Sigmoid activation)
```

**Total Architecture:** 784 → 128 → 10

### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Cross-Entropy |
| **Learning Rate** | 0.001 |
| **Batch Size** | 16 |
| **Epochs** | 2 |
| **Training Samples** | 100 (subset) |

### Test Results

**Test Suite Execution:** `test_mnist_training.cpp`

#### Test 1: Network Creation
- ✓ Network created successfully
- ✓ Optimizer set to Adam
- ✓ Loss set to Cross-Entropy
- ✓ Forward pass works correctly
- ✓ Output dimensions correct (1, 10)

#### Test 2: Batch Training Functionality
- **Initial Loss:** 0.5871
- **Final Loss (10 iterations):** 0.0000
- ✓ Loss decreased during training
- **Loss Reduction:** 100% (converged to near-zero)

#### Test 3: Accuracy Computation
- **Test Cases:** 5 samples
- **Correct Predictions:** 4 out of 5
- **Computed Accuracy:** 80.0%
- ✓ Accuracy computation verified correct

#### Test 4: MNIST Mini Training
- **Status:** Requires MNIST data files
- **Expected Behavior:**
  - Initial accuracy: ~10% (random guess)
  - Final accuracy: >20% (demonstrating learning)
  - Loss should decrease over 2 epochs

### Implementation Files

- **Test Suite:** `test_mnist_training.cpp` (lines 1-304)
- **Loader Test:** `test_mnist_loader.cpp` (lines 1-142)

---

## Experiment 3: Baseline Test (Synthetic Data)

### Network Architecture

```
Input Layer:    784 neurons  (ReLU activation)
                ↓
Hidden Layer:   64 neurons   (ReLU activation)
                ↓
Output Layer:   10 neurons   (Sigmoid activation)
```

**Total Architecture:** 784 → 64 → 10

### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Cross-Entropy |
| **Learning Rate** | 0.01 |
| **Batch Size** | 4 |
| **Training Iterations** | 10 |
| **Data Type** | Synthetic (simple patterns) |

### Results

- **Initial Loss:** 0.5871
- **Final Loss:** 0.0000
- **Loss Reduction:** 100%
- ✓ Demonstrates network can learn synthetic patterns

---

## Key Findings

### Architecture Insights

1. **Hidden Layer Size Impact:**
   - Larger networks (784→256→128→10) expected to achieve 85-92% accuracy
   - Smaller networks (784→128→10) suitable for quick testing
   - Very small networks (784→64→10) can still learn but with reduced capacity

2. **Activation Function Choice:**
   - ReLU in hidden layers prevents vanishing gradient
   - Sigmoid in output layer works well with cross-entropy loss
   - All architectures use same activation pattern

3. **Optimizer Performance:**
   - Adam optimizer shows excellent convergence
   - Loss can reach near-zero on synthetic data in 10 iterations
   - Learning rate of 0.001 is stable for MNIST

### Training Insights

1. **Batch Size:**
   - Batch size 32 for full training (good balance)
   - Batch size 16 for small experiments (faster iteration)
   - Smaller batches = noisier gradients but faster updates

2. **Learning Rate:**
   - 0.001 is standard for Adam on MNIST
   - 0.01 can work for very small synthetic tests
   - Adam adapts learning rate per parameter

3. **Convergence:**
   - Synthetic data: near-perfect convergence in 10 iterations
   - Real MNIST: expected 5-10 epochs for good performance
   - Loss decreases monotonically when properly configured

---

## Code Examples

### Loading MNIST Data

```cpp
#include "mnist_loader.h"

MNISTDataset<double> trainDataset;
if (!loadMNISTDataset<double>(
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    trainDataset)) {
    // Handle error
}
```

### Creating Network

```cpp
Network<double>* network = new Network<double>();

ILayer<double>* input = new Layer<double>(784, "Input", ActivationType::RELU);
ILayer<double>* hidden1 = new Layer<double>(256, "Hidden1", ActivationType::RELU);
ILayer<double>* hidden2 = new Layer<double>(128, "Hidden2", ActivationType::RELU);
ILayer<double>* output = new Layer<double>(10, "Output", ActivationType::SIGMOID);

network->setInputLayer(input);
network->connect(input, hidden1);
network->connect(hidden1, hidden2);
network->connect(hidden2, output);
network->setOutputLayer(output);

network->setOptimizerType(OptimizerType::ADAM);
network->setLossType(LossType::CROSS_ENTROPY);
network->init();
```

### Training Loop

```cpp
int batchSize = 32;
double learningRate = 0.001;

for (int epoch = 0; epoch < 10; epoch++) {
    // Batch training
    for (int batch = 0; batch < numBatches; batch++) {
        auto [batchImages, batchLabels] = getBatch(trainDataset,
                                                   batch * batchSize,
                                                   batchSize);
        network->trainBatch(batchImages, batchLabels, learningRate);
    }

    // Evaluate
    double accuracy = network->evaluateAccuracy(testImages, testLabels);
    double loss = network->evaluateLoss(testImages, testLabels);

    std::cout << "Epoch " << epoch
              << ": Accuracy=" << accuracy
              << "%, Loss=" << loss << std::endl;
}
```

---

## Reproducibility

### Environment

- **Language:** C++17
- **Build System:** CMake 3.10+
- **Compiler:** GCC 13.3.0
- **Dependencies:** OpenMP (optional, for parallelization)

### Build Commands

```bash
# Build test suite
make test_mnist_training

# Run tests
make run_test_mnist_training

# Build full trainer
make train_mnist

# Run training (with MNIST data present)
./build/train_mnist --epochs 10 --batch-size 32 --lr 0.001
```

### Data Download

```bash
# Download MNIST dataset
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Decompress
gunzip *.gz
```

---

## Future Experiments

### Planned Variations

1. **Architecture Experiments:**
   - [ ] Test deeper networks (4-5 hidden layers)
   - [ ] Test wider networks (512, 1024 neurons)
   - [ ] Test different activation functions (Leaky ReLU, Tanh)

2. **Optimizer Experiments:**
   - [ ] SGD with momentum
   - [ ] RMSprop
   - [ ] Compare convergence speeds

3. **Regularization Experiments:**
   - [ ] Add L2 weight regularization
   - [ ] Test dropout layers
   - [ ] Data augmentation (rotation, translation)

4. **Hyperparameter Tuning:**
   - [ ] Learning rate sweep: 0.0001, 0.001, 0.01
   - [ ] Batch size impact: 8, 16, 32, 64, 128
   - [ ] Hidden layer size: 64, 128, 256, 512

---

## References

- **MNIST Dataset:** LeCun, Y., Cortes, C., & Burges, C. (1998)
  - http://yann.lecun.com/exdb/mnist/

- **Adam Optimizer:** Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization."
  - https://arxiv.org/abs/1412.6980

- **Network Implementation:** `network.h`, `activation.h`, `optimizer.h`, `loss.h`

---

**Last Updated:** 2025-11-17
**Maintained By:** Network Team
**Version:** 1.0
