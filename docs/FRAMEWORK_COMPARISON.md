# Framework Comparison: PyTorch vs TensorFlow vs This Library

This document compares the syntax and usage patterns for key features implemented in this C++ neural network library against industry-standard frameworks PyTorch and TensorFlow.

---

## Table of Contents
- [Overview](#overview)
- [Critical Limitations](#critical-limitations)
- [Tensor Operations](#tensor-operations)
- [Convolutional Layers](#convolutional-layers)
- [Pooling Layers](#pooling-layers)
- [Training Flow](#training-flow)
- [Weight Initialization](#weight-initialization)
- [Summary](#summary)

---

## Overview

This library is a **CPU-only**, header-only C++ implementation designed for educational purposes and understanding deep learning fundamentals. While it shares conceptual similarities with PyTorch and TensorFlow, it lacks many production features.

### Feature Matrix

| Feature | PyTorch | TensorFlow | This Library |
|---------|---------|------------|--------------|
| Language | Python (C++ backend) | Python (C++ backend) | C++ (header-only) |
| GPU/CUDA Support | ✅ Yes | ✅ Yes | ❌ **NO** |
| Automatic Differentiation | ✅ Autograd | ✅ GradientTape | ⚠️ Manual |
| Dynamic Computation Graph | ✅ Yes | ✅ Eager mode | ❌ No |
| Production Ready | ✅ Yes | ✅ Yes | ❌ Educational |
| Distributed Training | ✅ Yes | ✅ Yes | ❌ No |
| Model Serialization | ✅ Yes | ✅ Yes | ❌ No |
| Pre-trained Models | ✅ torchvision | ✅ tf.keras.applications | ❌ No |

---

## Critical Limitations

### ⚠️ NO GPU/CUDA SUPPORT

**This library does NOT support GPU acceleration.** All computations run on CPU only.

| Capability | PyTorch | TensorFlow | This Library |
|------------|---------|------------|--------------|
| GPU Acceleration | ✅ CUDA, ROCm | ✅ CUDA, ROCm | ❌ **CPU ONLY** |
| Tensor Device | `.to('cuda')` | `tf.device('/GPU:0')` | ❌ N/A |
| Mixed Precision | ✅ AMP | ✅ mixed_precision | ❌ No |
| Multi-GPU | ✅ DataParallel, DDP | ✅ MirroredStrategy | ❌ No |
| Performance | ~100-1000x faster (GPU) | ~100-1000x faster (GPU) | Baseline (CPU) |

**For production workloads requiring GPU acceleration, use PyTorch or TensorFlow.**

### Other Missing Features
- No automatic differentiation (manual backward passes required)
- No dynamic computation graphs
- No model checkpointing/serialization
- No distributed training
- No quantization or optimization tools
- No pre-trained models
- Limited to basic architectures (MLP, CNN)

---

## Tensor Operations

### Creating Tensors

#### PyTorch
```python
import torch

# From values
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Zeros, ones, random
zeros = torch.zeros(2, 3, 4, 4)
ones = torch.ones(2, 3, 4, 4)
rand = torch.rand(2, 3, 4, 4)      # Uniform [0, 1)
randn = torch.randn(2, 3, 4, 4)    # Normal N(0, 1)

# Custom range
custom = torch.rand(2, 3, 4, 4) * 10.0  # Uniform [0, 10)

# Shape
print(x.shape)  # torch.Size([2, 2])
print(x.ndim)   # 2
```

#### TensorFlow
```python
import tensorflow as tf

# From values
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

# Zeros, ones, random
zeros = tf.zeros([2, 3, 4, 4])
ones = tf.ones([2, 3, 4, 4])
rand = tf.random.uniform([2, 3, 4, 4], 0, 1)      # Uniform [0, 1)
randn = tf.random.normal([2, 3, 4, 4], 0, 1)      # Normal N(0, 1)

# Custom range
custom = tf.random.uniform([2, 3, 4, 4], 0, 10)   # Uniform [0, 10)

# Shape
print(x.shape)  # TensorShape([2, 2])
print(len(x.shape))  # 2
```

#### This Library (C++)
```cpp
#include "tensor.h"

// From values (manual initialization required)
Tensor<float> x({2, 2});
x(0, 0) = 1; x(0, 1) = 2;
x(1, 0) = 3; x(1, 1) = 4;

// Zeros, ones, random
auto zeros = Tensor<float>::zeros({2, 3, 4, 4});
auto ones = Tensor<float>::ones({2, 3, 4, 4});
auto rand = Tensor<float>::random({2, 3, 4, 4}, 0.0f, 1.0f);    // Uniform [0, 1)
auto randn = Tensor<float>::randn({2, 3, 4, 4}, 0.0f, 1.0f);   // Normal N(0, 1)

// Custom range
auto custom = Tensor<float>::random({2, 3, 4, 4}, 0.0f, 10.0f); // Uniform [0, 10)

// Shape
auto shape = x.shape();     // std::vector<size_t>
size_t ndim = x.ndim();     // 2
```

**Key Differences:**
- Python frameworks use lists for shapes; C++ uses `std::vector<size_t>`
- C++ requires explicit template types (`<float>`)
- C++ indexing via `operator()`: `x(i, j, k, l)`

---

### Reshaping Tensors

#### PyTorch
```python
x = torch.randn(2, 3, 4, 4)

# Reshape
y = x.view(2, 3, 16)        # Must be contiguous
y = x.reshape(2, 3, 16)     # Works always (may copy)

# Squeeze/unsqueeze
z = x.unsqueeze(0)          # Add dimension: [1, 2, 3, 4, 4]
w = z.squeeze(0)            # Remove dimension: [2, 3, 4, 4]

# Transpose
t = x.transpose(2, 3)       # Swap dims 2 and 3
```

#### TensorFlow
```python
x = tf.random.normal([2, 3, 4, 4])

# Reshape
y = tf.reshape(x, [2, 3, 16])

# Expand/squeeze dims
z = tf.expand_dims(x, axis=0)      # Add dimension: [1, 2, 3, 4, 4]
w = tf.squeeze(z, axis=0)          # Remove dimension: [2, 3, 4, 4]

# Transpose
t = tf.transpose(x, perm=[0, 1, 3, 2])  # Swap dims 2 and 3
```

#### This Library (C++)
```cpp
Tensor<float> x = Tensor<float>::randn({2, 3, 4, 4}, 0.0f, 1.0f);

// Reshape
auto y = x.reshape({2, 3, 16});

// Squeeze/unsqueeze
auto z = x.unsqueeze(0);            // Add dimension: [1, 2, 3, 4, 4]
auto w = z.squeeze(0);              // Remove dimension: [2, 3, 4, 4]

// Transpose (2D only via Mat conversion)
// Note: Limited transpose support for 4D tensors
```

**Key Differences:**
- PyTorch has `view` (contiguous) vs `reshape` (may copy)
- TensorFlow uses explicit `axis` parameters
- This library has limited transpose support for high-dimensional tensors

---

### Element-wise Operations

#### PyTorch
```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# Arithmetic
c = a + b
c = a - b
c = a * b
c = a / b

# In-place
a += b
a *= 2.0

# Activation functions
relu = torch.relu(a)
sigmoid = torch.sigmoid(a)
tanh = torch.tanh(a)
```

#### TensorFlow
```python
a = tf.random.normal([2, 3])
b = tf.random.normal([2, 3])

# Arithmetic
c = a + b
c = a - b
c = a * b
c = a / b

# In-place not supported (tensors are immutable)
a = a + b
a = a * 2.0

# Activation functions
relu = tf.nn.relu(a)
sigmoid = tf.nn.sigmoid(a)
tanh = tf.nn.tanh(a)
```

#### This Library (C++)
```cpp
Tensor<float> a = Tensor<float>::randn({2, 3}, 0.0f, 1.0f);
Tensor<float> b = Tensor<float>::randn({2, 3}, 0.0f, 1.0f);

// Arithmetic
auto c = a + b;
auto c = a - b;
auto c = a * b;
auto c = a / b;

// Scalar operations
auto d = a * 2.0f;
auto d = a + 5.0f;

// Activation functions (via activation.h)
#include "activation.h"
// Applied during layer operations, not as standalone tensor ops
```

**Key Differences:**
- PyTorch and TensorFlow support in-place operations
- This library provides basic arithmetic operators
- Activations are applied in layer forward passes, not as standalone ops

---

## Convolutional Layers

### Layer Definition

#### PyTorch
```python
import torch.nn as nn

# Define Conv2D layer
conv = nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=5,      # Can be int or tuple (5, 5)
    stride=1,
    padding=2
)

# Multiple layers
model = nn.Sequential(
    nn.Conv2d(1, 32, 5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

# Forward pass
x = torch.randn(8, 3, 28, 28)  # [batch, channels, height, width]
output = conv(x)
print(output.shape)  # torch.Size([8, 32, 28, 28])
```

#### TensorFlow (Keras)
```python
import tensorflow as tf
from tensorflow.keras import layers

# Define Conv2D layer
conv = layers.Conv2D(
    filters=32,
    kernel_size=(5, 5),    # Must be tuple
    strides=(1, 1),
    padding='same',        # 'same' or 'valid'
    activation='relu'      # Can specify here
)

# Multiple layers
model = tf.keras.Sequential([
    layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2))
])

# Forward pass
x = tf.random.normal([8, 28, 28, 3])  # [batch, height, width, channels] - NHWC
output = conv(x)
print(output.shape)  # TensorShape([8, 28, 28, 32])
```

#### This Library (C++)
```cpp
#include "conv_layer.h"
using ml::ActivationType;

// Define Conv2D layer
Conv2D<float> conv(
    32,                          // out_channels
    5, 5,                        // kernel_height, kernel_width
    ActivationType::RELU,        // activation function
    1, 1,                        // stride_h, stride_w
    2, 2                         // pad_h, pad_w
);

// Initialize for input channels
conv.initialize(3);  // in_channels = 3

// Forward pass
Tensor<float> x = Tensor<float>::randn({8, 3, 28, 28}, 0.0f, 1.0f);  // [batch, channels, height, width]
auto output = conv.forward(x);
std::cout << "Output shape: " << output.shape(0) << "x" << output.shape(1)
          << "x" << output.shape(2) << "x" << output.shape(3) << std::endl;
// Output: 8x32x28x28
```

**Key Differences:**

| Aspect | PyTorch | TensorFlow | This Library |
|--------|---------|------------|--------------|
| Input format | NCHW | NHWC | NCHW |
| Kernel size | `int` or `tuple` | `tuple` only | Separate `int` params |
| Padding | `int` or `'same'` | `'same'` or `'valid'` | Explicit `int` values |
| Activation | Separate layer | Can specify in Conv2D | Required parameter |
| Initialization | Automatic | Automatic | Explicit `initialize()` |

---

### Convolution Parameters

#### PyTorch
```python
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

# Access parameters
print(conv.weight.shape)  # torch.Size([64, 3, 3, 3]) - [out, in, kh, kw]
print(conv.bias.shape)    # torch.Size([64])

# Manual initialization
nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')  # He
nn.init.xavier_normal_(conv.weight)                                       # Xavier
nn.init.zeros_(conv.bias)
```

#### TensorFlow (Keras)
```python
conv = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')

# Build layer to create weights
conv.build((None, 28, 28, 3))  # [batch, h, w, channels]

# Access parameters
print(conv.kernel.shape)  # TensorShape([3, 3, 3, 64]) - [kh, kw, in, out]
print(conv.bias.shape)    # TensorShape([64])

# Manual initialization
from tensorflow.keras.initializers import HeNormal, GlorotNormal
conv = layers.Conv2D(
    64, (3, 3),
    kernel_initializer=HeNormal(),      # He initialization
    bias_initializer='zeros'
)
```

#### This Library (C++)
```cpp
Conv2D<float> conv(64, 3, 3, ActivationType::RELU, 1, 1, 1, 1);
conv.initialize(3);  // in_channels

// Access parameters
auto& kernels = conv.getKernels();  // Shape: [64, 3, 3, 3] - [out, in, kh, kw]
auto& bias = conv.getBias();        // Shape: [64]

std::cout << "Kernel shape: " << kernels.shape(0) << "x" << kernels.shape(1)
          << "x" << kernels.shape(2) << "x" << kernels.shape(3) << std::endl;

// Initialization happens automatically in initialize()
// Uses He initialization for ReLU, Xavier for others
```

**Key Differences:**
- **Weight shapes differ**: PyTorch/This Library use `[out, in, kh, kw]`, TensorFlow uses `[kh, kw, in, out]`
- PyTorch/TensorFlow have manual init APIs; this library auto-initializes based on activation
- This library requires explicit `initialize(in_channels)` call

---

## Pooling Layers

### Max Pooling

#### PyTorch
```python
import torch.nn as nn

# Define MaxPool layer
pool = nn.MaxPool2d(
    kernel_size=2,    # Pool window size
    stride=2          # Stride (default: same as kernel_size)
)

# Forward pass
x = torch.randn(8, 32, 24, 24)
output = pool(x)
print(output.shape)  # torch.Size([8, 32, 12, 12])

# Overlapping pooling
pool_overlap = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
```

#### TensorFlow (Keras)
```python
from tensorflow.keras import layers

# Define MaxPool layer
pool = layers.MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'    # No padding
)

# Forward pass
x = tf.random.normal([8, 24, 24, 32])  # NHWC format
output = pool(x)
print(output.shape)  # TensorShape([8, 12, 12, 32])

# Overlapping pooling
pool_overlap = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')
```

#### This Library (C++)
```cpp
#include "pooling_layer.h"

// Define MaxPool layer
MaxPool2D<float> pool(2);  // Square pooling, stride = pool_size

// Explicit stride
MaxPool2D<float> pool_explicit(2, 2, 2, 2);  // pool_h, pool_w, stride_h, stride_w

// Forward pass
Tensor<float> x = Tensor<float>::randn({8, 32, 24, 24}, 0.0f, 1.0f);
auto output = pool.forward(x);
std::cout << "Output shape: " << output.shape(0) << "x" << output.shape(1)
          << "x" << output.shape(2) << "x" << output.shape(3) << std::endl;
// Output: 8x32x12x12

// Overlapping pooling (pool=3, stride=2)
MaxPool2D<float> pool_overlap(3, 3, 2, 2);
```

**Key Differences:**
- PyTorch uses `kernel_size`, TensorFlow uses `pool_size`
- This library requires separate height/width parameters
- Padding not yet implemented in this library

---

### Average Pooling

#### PyTorch
```python
pool = nn.AvgPool2d(kernel_size=2, stride=2)

x = torch.randn(8, 32, 24, 24)
output = pool(x)  # Shape: [8, 32, 12, 12]
```

#### TensorFlow (Keras)
```python
pool = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))

x = tf.random.normal([8, 24, 24, 32])
output = pool(x)  # Shape: [8, 12, 12, 32]
```

#### This Library (C++)
```cpp
AvgPool2D<float> pool(2);  // Square pooling

Tensor<float> x = Tensor<float>::randn({8, 32, 24, 24}, 0.0f, 1.0f);
auto output = pool.forward(x);  // Shape: [8, 32, 12, 12]
```

---

### Global Average Pooling

#### PyTorch
```python
# Option 1: AdaptiveAvgPool2d
gap = nn.AdaptiveAvgPool2d((1, 1))

x = torch.randn(8, 512, 7, 7)
output = gap(x)  # Shape: [8, 512, 1, 1]

# Option 2: Manual
output = x.mean(dim=[2, 3], keepdim=True)  # Same result
```

#### TensorFlow (Keras)
```python
gap = layers.GlobalAveragePooling2D()

x = tf.random.normal([8, 7, 7, 512])
output = gap(x)  # Shape: [8, 512] - flattened!

# To keep dims: use keepdims in reduce_mean
output = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # Shape: [8, 1, 1, 512]
```

#### This Library (C++)
```cpp
GlobalAvgPool2D<float> gap;

Tensor<float> x = Tensor<float>::randn({8, 512, 7, 7}, 0.0f, 1.0f);
auto output = gap.forward(x);  // Shape: [8, 512, 1, 1]
```

**Key Differences:**
- TensorFlow's `GlobalAveragePooling2D` flattens output; PyTorch/This Library keep spatial dims
- This library always outputs `[batch, channels, 1, 1]`

---

## Training Flow

### Forward and Backward Pass

#### PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
model = nn.Sequential(
    nn.Conv2d(1, 32, 5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 10)
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients (autograd!)
        optimizer.step()       # Update weights
```

#### TensorFlow (Keras)
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define model
model = models.Sequential([
    layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10)
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Training loop (high-level)
model.fit(train_dataset, epochs=10, batch_size=32)

# Manual training loop (low-level)
optimizer = tf.keras.optimizers.SGD(0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(10):
    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = model(batch_x, training=True)
            loss = loss_fn(batch_y, outputs)

        # Backward pass
        gradients = tape.gradient(loss, model.trainable_variables)  # Autograd!
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

#### This Library (C++)
```cpp
#include "conv_layer.h"
#include "pooling_layer.h"
#include "tensor.h"

// Define layers manually
Conv2D<float> conv1(32, 5, 5, ActivationType::RELU, 1, 1, 2, 2);
conv1.initialize(1);  // 1 input channel (grayscale)

MaxPool2D<float> pool1(2);

Conv2D<float> conv2(64, 5, 5, ActivationType::RELU, 1, 1, 2, 2);
conv2.initialize(32);

MaxPool2D<float> pool2(2);

// Training loop
float learning_rate = 0.01f;

for (int epoch = 0; epoch < 10; ++epoch) {
    for (auto& [batch_x, batch_y] : dataloader) {
        // Forward pass (manual layer chaining)
        auto h1 = conv1.forward(batch_x);      // [batch, 1, 28, 28] -> [batch, 32, 28, 28]
        auto h2 = pool1.forward(h1);           // [batch, 32, 28, 28] -> [batch, 32, 14, 14]
        auto h3 = conv2.forward(h2);           // [batch, 32, 14, 14] -> [batch, 64, 14, 14]
        auto h4 = pool2.forward(h3);           // [batch, 64, 14, 14] -> [batch, 64, 7, 7]

        // ... flatten and fully connected layers ...

        // Compute loss (manual)
        auto loss_grad = compute_loss_gradient(output, batch_y);

        // Backward pass (manual gradient propagation!)
        auto d_h4 = /* gradient from FC layers */;
        auto d_h3 = pool2.backward(d_h4);
        auto d_h2 = conv2.backward(d_h3);
        auto d_h1 = pool1.backward(d_h2);
        auto d_input = conv1.backward(d_h1);

        // Update weights (manual)
        conv1.updateWeights(learning_rate);
        conv2.updateWeights(learning_rate);
    }
}
```

**Key Differences:**

| Aspect | PyTorch | TensorFlow | This Library |
|--------|---------|------------|--------------|
| Autograd | ✅ Automatic | ✅ GradientTape | ❌ **Manual backprop** |
| Layer chaining | `nn.Sequential` | `models.Sequential` | Manual forward calls |
| Gradient computation | `loss.backward()` | `tape.gradient()` | Manual `backward()` calls |
| Weight updates | `optimizer.step()` | `apply_gradients()` | Manual `updateWeights()` |
| Ease of use | Very easy | Very easy | Complex, error-prone |

**This library requires manual gradient propagation through each layer in reverse order!**

---

## Weight Initialization

### He Initialization (for ReLU)

#### PyTorch
```python
import torch.nn as nn
import torch.nn.init as init

conv = nn.Conv2d(3, 64, 3)
init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
init.zeros_(conv.bias)

# Or use default initialization (already good for most cases)
conv = nn.Conv2d(3, 64, 3)  # Uses kaiming_uniform_ by default
```

#### TensorFlow (Keras)
```python
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal

conv = layers.Conv2D(
    64, (3, 3),
    kernel_initializer=HeNormal(),
    bias_initializer='zeros'
)

# Or use string alias
conv = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal')
```

#### This Library (C++)
```cpp
// He initialization happens automatically for ReLU activations
Conv2D<float> conv(64, 3, 3, ActivationType::RELU, 1, 1, 1, 1);
conv.initialize(3);  // Automatically uses He init for ReLU

// Manual He initialization formula used internally:
// std = sqrt(2.0 / fan_in)
// where fan_in = in_channels * kernel_h * kernel_w
```

---

### Xavier Initialization (for Tanh/Sigmoid)

#### PyTorch
```python
import torch.nn as nn
import torch.nn.init as init

conv = nn.Conv2d(3, 64, 3)
init.xavier_normal_(conv.weight, gain=1.0)
init.zeros_(conv.bias)
```

#### TensorFlow (Keras)
```python
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotNormal

conv = layers.Conv2D(
    64, (3, 3),
    kernel_initializer=GlorotNormal(),  # Xavier = Glorot
    bias_initializer='zeros'
)

# Or use string alias
conv = layers.Conv2D(64, (3, 3), kernel_initializer='glorot_normal')
```

#### This Library (C++)
```cpp
// Xavier initialization happens automatically for non-ReLU activations
Conv2D<float> conv(64, 3, 3, ActivationType::TANH, 1, 1, 1, 1);
conv.initialize(3);  // Automatically uses Xavier init for Tanh

// Manual Xavier initialization formula used internally:
// std = sqrt(2.0 / (fan_in + fan_out))
// where fan_in = in_channels * kernel_h * kernel_w
//       fan_out = out_channels * kernel_h * kernel_w
```

**Key Differences:**
- PyTorch/TensorFlow allow manual initialization control
- This library auto-selects initialization based on activation function
- PyTorch calls it "Kaiming", TensorFlow calls it "He" (same algorithm)
- TensorFlow calls Xavier "Glorot" (after the paper's author)

---

## Summary

### When to Use Each Framework

#### Use **PyTorch** if you need:
- ✅ GPU acceleration (CUDA)
- ✅ Rapid prototyping and research
- ✅ Dynamic computation graphs
- ✅ Strong ecosystem (torchvision, etc.)
- ✅ Python-first API
- ✅ Production deployment

#### Use **TensorFlow** if you need:
- ✅ GPU acceleration (CUDA)
- ✅ Production deployment at scale
- ✅ TensorFlow Lite for mobile/embedded
- ✅ TensorFlow.js for browser deployment
- ✅ Robust serving infrastructure (TF Serving)
- ✅ Industry-standard framework

#### Use **This Library** if you want:
- 📚 Educational understanding of CNNs
- 🔧 Learning implementation details
- 🧪 Small-scale CPU experiments
- 📖 Reading C++ deep learning code
- ⚠️ **NOT for production use**
- ⚠️ **NO GPU support**

---

## Quick Syntax Comparison Table

| Operation | PyTorch | TensorFlow | This Library (C++) |
|-----------|---------|------------|-------------------|
| **Create tensor** | `torch.randn(2, 3)` | `tf.random.normal([2, 3])` | `Tensor<float>::randn({2, 3}, 0, 1)` |
| **Conv2D** | `nn.Conv2d(3, 32, 5)` | `layers.Conv2D(32, (5,5))` | `Conv2D<float>(32, 5, 5, ...)` |
| **MaxPool** | `nn.MaxPool2d(2)` | `layers.MaxPool2D((2,2))` | `MaxPool2D<float>(2)` |
| **Forward pass** | `model(x)` | `model(x)` | `layer.forward(x)` |
| **Backward pass** | `loss.backward()` | `tape.gradient(loss, vars)` | Manual `layer.backward(d_out)` |
| **Update weights** | `optimizer.step()` | `optimizer.apply_gradients()` | `layer.updateWeights(lr)` |
| **Data format** | NCHW (default) | NHWC (default) | NCHW only |
| **GPU support** | ✅ `x.to('cuda')` | ✅ `with tf.device('/GPU:0')` | ❌ **CPU ONLY** |

---

## Conclusion

This C++ library provides a **foundational understanding** of how CNNs work under the hood. However, it **lacks critical production features**:

- ❌ **NO GPU/CUDA SUPPORT** - Orders of magnitude slower than PyTorch/TensorFlow on GPU
- ❌ No automatic differentiation
- ❌ No dynamic computation graphs
- ❌ No pre-trained models
- ❌ No distributed training
- ❌ No model serialization

**For any real-world deep learning project, use PyTorch or TensorFlow.** This library is strictly for educational purposes to understand the implementation details of CNNs in C++.

---

## Additional Resources

### PyTorch
- Official Docs: https://pytorch.org/docs/
- Tutorials: https://pytorch.org/tutorials/
- CUDA Support: https://pytorch.org/docs/stable/notes/cuda.html

### TensorFlow
- Official Docs: https://www.tensorflow.org/api_docs
- Tutorials: https://www.tensorflow.org/tutorials
- GPU Support: https://www.tensorflow.org/guide/gpu

### This Library
- See `docs/ARCHITECTURE_DESIGN.md` for implementation details
- See `docs/CNN_COMPATIBILITY.md` for CNN concepts
- See `docs/IMPLEMENTATION_PROGRESS.md` for status tracking

---

**Last Updated:** 2025-11-17
