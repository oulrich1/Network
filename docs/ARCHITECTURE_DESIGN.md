# Pluggable Neural Network Architecture Design

## Executive Summary

This document describes the design for extending the existing feedforward neural network library to support Convolutional Neural Networks (CNNs) and future Recurrent Neural Networks (RNNs) through a pluggable architecture pattern based on dependency injection principles.

**Design Philosophy**: Minimize breaking changes, maximize extensibility, maintain performance.

---

## 1. Core Design Principles

### 1.1 Dependency Injection for Neural Architectures

We treat different layer types as **injectable dependencies** that conform to a common interface:

```cpp
// The contract all layers must fulfill
template <typename T>
class ILayer {
    virtual void feed(Mat<T> in, int epoch) = 0;
    virtual void backprop() = 0;
    virtual void init() = 0;
    // ... other interface methods
};
```

**Benefits**:
- **Open/Closed Principle**: Open for extension (new layer types), closed for modification (core network logic unchanged)
- **Testability**: Each layer type can be tested in isolation
- **Composability**: Layers can be mixed and matched (Dense → Conv → Dense)
- **Maintainability**: Clear separation of concerns

### 1.2 Layer Type Hierarchy

```
INode<T>                          [Base interface with visit tracking]
  └─ ILayer<T>                    [Abstract layer with graph connectivity]
       ├─ DenseLayer<T>           [Fully-connected layer - existing Layer<T>]
       ├─ ConvLayer<T>            [NEW: Convolutional layer]
       │    ├─ Conv2D<T>          [2D convolution]
       │    └─ Conv1D<T>          [1D convolution - future]
       ├─ PoolingLayer<T>         [NEW: Pooling layer]
       │    ├─ MaxPool2D<T>       [Max pooling]
       │    └─ AvgPool2D<T>       [Average pooling]
       └─ RecurrentLayer<T>       [FUTURE: RNN family]
            ├─ SimpleRNN<T>
            ├─ LSTM<T>
            └─ GRU<T>
```

### 1.3 Backward Compatibility Strategy

**No Breaking Changes**:
- Existing `Layer<T>` becomes `DenseLayer<T>` internally
- Provide type alias: `template <typename T> using Layer = DenseLayer<T>;`
- All existing code continues to compile and run

---

## 2. Key Architectural Decisions

### Decision 1: Tensor Representation

**Problem**: Current `Mat<T>` is 2D only. CNNs need 3D/4D tensors (batch, channels, height, width).

**Solution**: Add `Tensor<T>` alongside `Mat<T>` (non-breaking).

```cpp
template <typename T>
class Tensor {
public:
    std::vector<size_t> shape;  // e.g., [batch, channels, height, width]
    std::shared_ptr<T> data;    // Contiguous memory

    // Factory methods
    static Tensor<T> fromMat(const Mat<T>& mat, std::vector<size_t> shape);
    Mat<T> toMat() const;  // Flatten to 2D

    // Indexing
    T& operator()(size_t b, size_t c, size_t h, size_t w);

    // Operations
    Tensor<T> reshape(std::vector<size_t> newShape);
    Tensor<T> transpose(std::vector<int> axes);
};
```

**Rationale**:
- Dense layers continue using `Mat<T>` (no performance regression)
- CNN layers use `Tensor<T>` internally
- Conversion between Mat/Tensor at layer boundaries
- Future-proof for RNN (3D: batch, time, features)

### Decision 2: Data Flow in Mixed Networks

**Challenge**: How does data flow from DenseLayer (outputs Mat) to ConvLayer (expects Tensor)?

**Solution**: Each layer handles input shape transformations.

```cpp
// Example: Dense → Conv → Dense network
Network<float> net;
auto* dense1 = net.addLayer(new DenseLayer<float>(784, RELU));  // MNIST input
auto* conv1  = net.addLayer(new Conv2D<float>(32, 3, 3, RELU)); // Auto-reshape 784→28x28x1
auto* pool1  = net.addLayer(new MaxPool2D<float>(2, 2));        // 28x28→14x14
auto* dense2 = net.addLayer(new DenseLayer<float>(10, SOFTMAX)); // Auto-flatten

// Each layer knows its expected input/output format
conv1->setInputShape({28, 28, 1});  // Explicit if needed
```

**Layer Responsibility**:
- Each layer's `feed()` method handles input format conversion
- Store metadata: `mInputShape`, `mOutputShape`
- Network validates shape compatibility during `init()`

### Decision 3: Convolution Implementation Strategy

**Approach**: Use **im2col** (image-to-column) transformation.

**Why im2col?**:
- Converts convolution to matrix multiplication (leverages existing `Mat::Mult`)
- Used by Caffe, PyTorch, TensorFlow for CPU convolution
- Simpler to implement than direct convolution
- Easy to optimize with existing BLAS/OpenMP

**Algorithm**:
```cpp
// Forward pass
Tensor<T> ConvLayer::feed(const Tensor<T>& input) {
    // input: [batch, in_channels, in_h, in_w]
    // kernel: [out_channels, in_channels, kh, kw]

    Mat<T> col = im2col(input);  // [batch*out_h*out_w, in_ch*kh*kw]
    Mat<T> kernel_mat = reshapeKernel();  // [out_ch, in_ch*kh*kw]
    Mat<T> output_mat = Mult(col, kernel_mat.T());  // [batch*out_h*out_w, out_ch]

    return col2im(output_mat);  // [batch, out_channels, out_h, out_w]
}

// Backward pass
void ConvLayer::backprop() {
    // d_col = d_output via col2im
    // d_kernel = d_col^T * col (gradient for weights)
    // d_input = col2im(d_col * kernel) (gradient for previous layer)
}
```

**Trade-off**:
- **Pro**: Simple, correct, maintainable
- **Con**: Extra memory (2x input size)
- **Future**: Can optimize with Winograd/FFT later

### Decision 4: Parameter Storage

**Current**: `map<ILayer<T>*, Mat<T>> weights` (per-connection weights)

**CNN Challenge**: Convolution has shared parameters (kernels), not per-connection.

**Solution**: Move parameter storage into each layer.

```cpp
class ILayer<T> {
protected:
    // NEW: Each layer owns its parameters
    std::vector<Mat<T>> mParameters;      // Weight matrices/tensors
    std::vector<Mat<T>> mParameterGrads;  // Gradient accumulation

public:
    virtual std::vector<Mat<T>>& getParameters() { return mParameters; }
    virtual void updateParameters(IOptimizer<T>* optimizer);
};

// Dense layer: mParameters = {weights, bias}
// Conv layer: mParameters = {kernels, bias}
// RNN layer: mParameters = {W_input, W_hidden, W_output, bias}
```

**Migration Path**:
- DenseLayer wraps existing `weights` map in `mParameters`
- Optimizer updates via `layer->updateParameters(optimizer)`
- Gradual migration from Network-managed weights to layer-managed

### Decision 5: Batch Processing

**Current**: Single-sample processing (loops externally)

**Future**: Native batch support for performance

**Staged Approach**:
1. **Phase 1 (Now)**: Keep single-sample interface
   - CNN layers process batch dimension internally
   - Input: `[1, channels, height, width]` (batch=1)
   - Easier to debug, maintains compatibility

2. **Phase 2 (Later)**: Add batch API
   ```cpp
   Mat<T> Network::feed(const Mat<T>& in);           // Single sample
   Tensor<T> Network::feedBatch(const Tensor<T>& in); // Batch support
   ```

**Rationale**: Incremental complexity, test correctness before optimizing.

---

## 3. CNN vs ANN: Core Differences

### 3.1 Mathematical Differences

| Aspect | Dense/ANN | Convolutional/CNN |
|--------|-----------|-------------------|
| **Operation** | Matrix multiplication: `y = Wx + b` | Convolution: `y[i,j] = Σ w[m,n] * x[i+m, j+n] + b` |
| **Parameters** | Fully connected: O(n_in × n_out) | Shared kernels: O(k_h × k_w × c_in × c_out) |
| **Input** | 1D vector (flattened) | 2D/3D spatial (preserves structure) |
| **Output** | 1D vector | 2D/3D feature maps |
| **Connectivity** | Every input → every output | Local receptive fields |
| **Translation** | Not invariant | Translation equivariant |

### 3.2 Structural Differences

**Dense Layer**:
```
Input: [784] (28×28 image flattened)
Weights: [784, 128]
Output: [128]
Parameters: 100,352
```

**Conv Layer**:
```
Input: [1, 28, 28] (channels, height, width)
Kernel: [32, 1, 3, 3] (out_ch, in_ch, kh, kw)
Output: [32, 26, 26] (with valid padding)
Parameters: 32 × (1×3×3 + 1) = 320
```

**Parameter Efficiency**: 100,352 vs 320 (313x reduction!)

### 3.3 Implementation Requirements

**Dense Layer Needs**:
- Weight matrix initialization (Xavier/He)
- Activation function
- Bias term
- Forward: `σ(Wx + b)`
- Backward: `δ = W^T e ⊙ σ'(z)`

**Conv Layer Needs**:
- Kernel initialization (He for ReLU)
- Stride, padding, dilation parameters
- im2col/col2im utilities
- Forward: `σ(conv(x, w) + b)`
- Backward:
  - Weight gradient: `∇w = conv(x, δ)`
  - Input gradient: `∇x = conv_transpose(δ, w)`

**Pooling Layer Needs**:
- No trainable parameters
- Forward: Downsample (max or average)
- Backward: Upsample gradient (route to max indices or spread evenly)

---

## 4. Required Code Changes

### 4.1 New Files to Create

```
Network/
├── tensor.h              # NEW: N-dimensional tensor class
├── conv_layer.h          # NEW: Convolutional layer
├── pooling_layer.h       # NEW: Pooling layers
├── dense_layer.h         # NEW: Refactored from Layer<T>
├── layer_factory.h       # NEW: Factory for creating layers
├── im2col.h              # NEW: Convolution utilities
└── docs/
    ├── ARCHITECTURE_DESIGN.md    # This file
    └── CNN_COMPATIBILITY.md      # User guide
```

### 4.2 Modified Files

```
network.h
  - Add: virtual void updateParameters(IOptimizer<T>*) to ILayer
  - Add: getParameters() / setParameters() methods
  - Modify: Network<T> to handle different layer types
  - Add: Shape validation in init()

activation.h
  - Already supports all needed activations (ReLU, Softmax, etc.)
  - No changes required ✓

optimizer.h
  - Already abstracted via IOptimizer
  - Works with new parameter storage
  - No changes required ✓

Matrix/matrix.h
  - Add: Transpose() for non-square matrices (if missing)
  - Add: Pad() for convolution padding
  - Consider: Move toward expression templates (future optimization)
```

### 4.3 Backward Compatibility Layer

```cpp
// In network.h - add at end of file
#ifndef NETWORK_NO_COMPAT
template <typename T>
using Layer = DenseLayer<T>;  // Alias for existing code
#endif
```

Users can opt-out with: `#define NETWORK_NO_COMPAT`

---

## 5. Testing Strategy

### 5.1 Unit Tests for New Components

```cpp
test_tensor.cpp
  - Tensor creation, indexing, reshape
  - Conversions: Mat ↔ Tensor
  - Memory safety (shared_ptr semantics)

test_im2col.cpp
  - Correctness: im2col → col2im should be identity
  - Edge cases: padding, stride, dilation
  - Performance: benchmark against naive loops

test_conv_layer.cpp
  - Forward pass: compare against numpy.convolve
  - Backward pass: numerical gradient checking
  - Different parameters: stride, padding, channels

test_pooling_layer.cpp
  - MaxPool: verify max selection
  - AvgPool: verify averaging
  - Backward: gradient routing

test_cnn_network.cpp
  - Simple CNN: Conv → Pool → Dense
  - MNIST training: achieve >95% accuracy
  - Save/load: serialization of CNN models
```

### 5.2 Integration Tests

```cpp
test_mixed_architecture.cpp
  - Dense → Conv → Dense
  - Conv → Conv → Pool → Conv → Dense (deeper network)
  - Verify gradient flow across layer type boundaries

test_gradient_flow.cpp
  - Numerical vs analytical gradients
  - For each layer type in isolation
  - For full CNN networks
```

### 5.3 Regression Tests

```cpp
test_backward_compatibility.cpp
  - All existing tests should pass unchanged
  - Verify Layer<T> alias works
  - Verify existing saved models load correctly
```

---

## 6. Implementation Phases

### Phase 1: Foundation (Current Sprint)
- [x] Document architecture design
- [ ] Implement `Tensor<T>` class
- [ ] Implement `im2col` / `col2im` utilities
- [ ] Write tests for tensor and im2col
- [ ] Update documentation

### Phase 2: Convolutional Layers
- [ ] Implement `Conv2D<T>` layer
- [ ] Implement `MaxPool2D<T>` and `AvgPool2D<T>`
- [ ] Write unit tests for layers
- [ ] Numerical gradient checking

### Phase 3: Integration
- [ ] Refactor `Layer<T>` → `DenseLayer<T>`
- [ ] Update `Network<T>` for layer-owned parameters
- [ ] Implement layer factory pattern
- [ ] Integration tests

### Phase 4: Validation
- [ ] Train simple CNN on MNIST
- [ ] Compare accuracy vs TensorFlow/PyTorch
- [ ] Performance benchmarking
- [ ] Documentation and examples

### Phase 5: RNN Support (Future)
- [ ] Implement `SimpleRNN<T>`, `LSTM<T>`, `GRU<T>`
- [ ] Time-series data support
- [ ] Backpropagation through time (BPTT)

---

## 7. API Design Examples

### 7.1 Building a CNN (Declarative Style)

```cpp
// Example: LeNet-5 for MNIST
Network<float>* buildLeNet5() {
    auto* net = new Network<float>();

    // Input: 28x28x1
    auto* conv1 = new Conv2D<float>(6, 5, 5, RELU);    // → 24x24x6
    auto* pool1 = new MaxPool2D<float>(2, 2);          // → 12x12x6
    auto* conv2 = new Conv2D<float>(16, 5, 5, RELU);   // → 8x8x16
    auto* pool2 = new MaxPool2D<float>(2, 2);          // → 4x4x16
    auto* dense1 = new DenseLayer<float>(120, RELU);   // Flatten → 120
    auto* dense2 = new DenseLayer<float>(84, RELU);    // → 84
    auto* output = new DenseLayer<float>(10, SOFTMAX); // → 10

    net->setInputLayer(conv1);
    net->connect(conv1, pool1);
    net->connect(pool1, conv2);
    net->connect(conv2, pool2);
    net->connect(pool2, dense1);  // Auto-flatten
    net->connect(dense1, dense2);
    net->connect(dense2, output);
    net->setOutputLayer(output);

    return net;
}
```

### 7.2 Building a CNN (Builder Pattern - Future)

```cpp
auto net = NetworkBuilder<float>()
    .input({28, 28, 1})
    .conv2d(6, 5, 5, RELU)
    .maxpool(2, 2)
    .conv2d(16, 5, 5, RELU)
    .maxpool(2, 2)
    .flatten()
    .dense(120, RELU)
    .dense(84, RELU)
    .dense(10, SOFTMAX)
    .build();
```

### 7.3 Mixed Architecture

```cpp
// Custom architecture: Dense → Conv → Dense
auto* net = new Network<float>();

auto* dense1 = new DenseLayer<float>(256, RELU);
dense1->setOutputShape({16, 16, 1});  // Reshape for conv

auto* conv = new Conv2D<float>(8, 3, 3, RELU);
conv->setPadding(1);  // Preserve spatial dimensions

auto* pool = new MaxPool2D<float>(2, 2);  // → 8x8x8

auto* dense2 = new DenseLayer<float>(10, SOFTMAX);

net->connect(dense1, conv);
net->connect(conv, pool);
net->connect(pool, dense2);  // Auto-flatten 8x8x8 → 512
```

---

## 8. Performance Considerations

### 8.1 Memory Optimization

**im2col Memory**:
- Overhead: ~2x input size for column matrix
- Mitigation: Tile large images, reuse buffers

**Tensor Storage**:
- Use `std::shared_ptr` for copy-on-write semantics
- Avoid unnecessary copies in layer graph

### 8.2 Computation Optimization

**Immediate** (Phase 1-4):
- Leverage existing OpenMP in `Mat::Mult`
- im2col benefits from BLAS if linked

**Future** (Phase 5+):
- Winograd convolution (2.25x speedup for 3×3 kernels)
- FFT-based convolution for large kernels
- Strassen algorithm for large matrix multiplications
- GPU support (CUDA/OpenCL)

### 8.3 Cache Efficiency

- im2col improves cache locality (vs direct convolution)
- NCHW format (channels, height, width) for better vectorization
- Consider NHWC format for specific hardware (e.g., Intel CPUs)

---

## 9. Documentation Deliverables

### 9.1 For Users

1. **CNN_COMPATIBILITY.md**
   - What's different between Dense and Conv layers
   - When to use CNN vs Dense
   - Migration guide for existing code
   - Example architectures

2. **TUTORIAL_CNN.md**
   - Step-by-step LeNet-5 implementation
   - Training on MNIST
   - Hyperparameter tuning

3. **API_REFERENCE.md**
   - All layer types and their parameters
   - Tensor API
   - Network building methods

### 9.2 For Developers

1. **ARCHITECTURE_DESIGN.md** (this file)
   - Design rationale
   - Implementation details
   - Extension guide

2. **CONTRIBUTING.md**
   - How to add new layer types
   - Testing requirements
   - Code style guide

---

## 10. Success Criteria

### Functional Requirements
- ✓ CNN layers (Conv2D, MaxPool2D, AvgPool2D) implemented
- ✓ Backward compatibility: all existing tests pass
- ✓ Gradient correctness: numerical gradient checks pass
- ✓ MNIST CNN accuracy: >95% in <10 epochs

### Non-Functional Requirements
- ✓ Code coverage: >80% for new components
- ✓ Performance: CNN training within 2x of PyTorch CPU
- ✓ Documentation: all public APIs documented
- ✓ Maintainability: no files >500 lines

### Extensibility Requirements
- ✓ Adding new layer type requires <200 lines of code
- ✓ RNN support can be added without modifying existing layers
- ✓ External users can create custom layers

---

## 11. Future Roadmap

### Short Term (1-2 months)
- Batch normalization layer
- Dropout layer
- Data augmentation utilities
- More activation functions (Swish, Mish)

### Medium Term (3-6 months)
- RNN family (SimpleRNN, LSTM, GRU)
- 1D and 3D convolutions
- Attention mechanisms
- Model zoo (pre-trained models)

### Long Term (6-12 months)
- GPU acceleration (CUDA)
- Automatic differentiation (replace manual backprop)
- Dynamic computation graphs
- Distributed training

---

## 12. References

**Academic Papers**:
- LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition" (LeNet-5)
- He et al. (2015) - "Delving Deep into Rectifiers" (He initialization for CNNs)
- Chellapilla et al. (2006) - "High Performance Convolutional Neural Networks for Document Processing" (im2col)

**Implementation References**:
- Caffe: im2col implementation
- PyTorch: Conv2d backward pass
- CS231n: Stanford CNN course notes

**Design Patterns**:
- Gang of Four: Strategy, Factory, Composite patterns
- Dependency Injection in C++ (modern practices)

---

## Appendix A: Design Alternatives Considered

### Alternative 1: Separate CNN Library
**Rejected**: Would fragment the codebase, users want unified API

### Alternative 2: Direct Convolution (no im2col)
**Rejected**: More complex, harder to optimize, reinvents wheel

### Alternative 3: Template Specialization for Layer Types
```cpp
template <LayerType TYPE>
class Layer { ... };
```
**Rejected**: Less flexible than polymorphism, hard to extend

### Alternative 4: Python Bindings (use existing frameworks)
**Rejected**: Defeats purpose of C++ header-only library

---

## Appendix B: Migration Checklist for Existing Code

For users with existing codebases:

- [ ] Include new headers: `#include "tensor.h"`, `#include "conv_layer.h"`
- [ ] (Optional) Replace `Layer<T>` with `DenseLayer<T>` explicitly
- [ ] Rebuild and run existing tests - should pass unchanged
- [ ] Start using CNN layers in new models
- [ ] Update optimizer initialization if using custom optimizers
- [ ] Retrain models to generate new checkpoints (format may change)

**Estimated Migration Time**: <1 hour for typical project

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Authors**: AI Assistant (Claude) + Human Guidance
**Status**: Design Phase - Ready for Implementation
