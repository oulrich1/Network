# CNN Compatibility Guide

## Overview

This document describes the core differences between traditional Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs), and the changes required to support both in this library through a pluggable architecture.

---

## Why CNNs?

### Problem with Dense Networks for Images

Consider processing a 28×28 grayscale image (MNIST):

**Dense/ANN Approach**:
```cpp
// Flatten image to 1D vector
Input: [784]  (28 × 28 = 784)
Hidden: [128]
Weights: 784 × 128 = 100,352 parameters

Problems:
❌ Loses spatial structure (pixel neighborhoods destroyed)
❌ Huge parameter count (overfitting risk)
❌ Not translation invariant (cat in corner ≠ cat in center)
❌ Scales poorly (224×224 RGB image = 150K inputs!)
```

**Convolutional/CNN Approach**:
```cpp
// Preserve 2D spatial structure
Input: [1, 28, 28]  (channels, height, width)
Conv: 32 filters of 3×3
Weights: 32 × (1×3×3 + bias) = 320 parameters

Benefits:
✅ Preserves spatial relationships
✅ Parameter sharing (same filter across image)
✅ Translation equivariant (detects features anywhere)
✅ Hierarchical feature learning (edges → shapes → objects)
```

**Parameter Reduction**: 100,352 → 320 (313× fewer parameters!)

---

## Core Differences

### 1. Mathematical Operations

#### Dense Layer (Fully Connected)
```
Operation: y = σ(Wx + b)
  - Matrix multiplication
  - Every input connects to every output
  - No spatial awareness

Example:
Input:  x ∈ ℝ^784
Weight: W ∈ ℝ^(128 × 784)
Bias:   b ∈ ℝ^128
Output: y ∈ ℝ^128
```

#### Convolutional Layer
```
Operation: y[i,j] = σ(Σ_m Σ_n w[m,n] · x[i+m, j+n] + b)
  - Discrete convolution (cross-correlation)
  - Local receptive fields (kernel size)
  - Spatial relationships preserved

Example:
Input:  x ∈ ℝ^(1 × 28 × 28)
Kernel: w ∈ ℝ^(32 × 1 × 3 × 3)  (32 filters, 3×3 each)
Bias:   b ∈ ℝ^32
Output: y ∈ ℝ^(32 × 26 × 26)  (with valid padding)
```

### 2. Data Representation

| Aspect | Dense/ANN | Convolutional/CNN |
|--------|-----------|-------------------|
| **Input Shape** | 1D vector `[features]` | 3D tensor `[channels, height, width]` |
| **Output Shape** | 1D vector `[outputs]` | 3D tensor `[filters, out_h, out_w]` |
| **Internal Storage** | `Mat<T>` (2D matrix) | `Tensor<T>` (N-D tensor) |
| **Spatial Structure** | Lost (flattened) | Preserved |

### 3. Parameter Structure

#### Dense Layer Parameters
```cpp
class DenseLayer {
    Mat<T> weights;  // [input_size, output_size]
    Mat<T> bias;     // [output_size, 1]
};

// Unique weights for every connection
Total params = (input_size + 1) × output_size
```

#### Convolutional Layer Parameters
```cpp
class ConvLayer {
    Tensor<T> kernels;  // [out_channels, in_channels, kernel_h, kernel_w]
    Tensor<T> bias;     // [out_channels]
};

// Shared weights (parameter sharing)
Total params = out_channels × (in_channels × kernel_h × kernel_w + 1)
```

**Key Difference**: CNNs use **parameter sharing** - the same filter/kernel is applied across the entire spatial dimension.

### 4. Connectivity Patterns

```
Dense Layer (Fully Connected):
Input neurons:  [a, b, c, d, e, f]
Output neurons: [x, y]

Every input connects to every output:
x = w1a + w2b + w3c + w4d + w5e + w6f + bias
y = w7a + w8b + w9c + w10d + w11e + w12f + bias

12 unique weights (plus 2 biases)
```

```
Convolutional Layer (Local Connectivity):
Input (1D for simplicity): [a, b, c, d, e, f]
Kernel: [w1, w2, w3]  (size 3)
Stride: 1

Output[0] = w1·a + w2·b + w3·c + bias
Output[1] = w1·b + w2·c + w3·d + bias  (same w1, w2, w3!)
Output[2] = w1·c + w2·d + w3·e + bias
Output[3] = w1·d + w2·e + w3·f + bias

3 weights (plus 1 bias) - shared across all positions
```

### 5. Forward Propagation

#### Dense Layer
```cpp
Mat<T> DenseLayer::feed(const Mat<T>& input) {
    // input: [batch_size, input_dim]
    Mat<T> z = Mult(input, weights) + bias;  // Linear transform
    Mat<T> a = Activate(z, activationType);  // Activation
    return a;
}

Complexity: O(batch × input_dim × output_dim)
```

#### Convolutional Layer
```cpp
Tensor<T> ConvLayer::feed(const Tensor<T>& input) {
    // input: [batch, in_channels, height, width]

    // Convert convolution to matrix multiplication via im2col
    Mat<T> col = im2col(input, kernel_h, kernel_w, stride, padding);
    // col: [batch × out_h × out_w, in_channels × kernel_h × kernel_w]

    Mat<T> kernel_mat = reshapeKernels();
    // kernel_mat: [out_channels, in_channels × kernel_h × kernel_w]

    Mat<T> output_mat = Mult(col, kernel_mat.T()) + bias;
    // output_mat: [batch × out_h × out_w, out_channels]

    Tensor<T> output = col2im(output_mat, batch, out_channels, out_h, out_w);
    // output: [batch, out_channels, out_h, out_w]

    return Activate(output, activationType);
}

Complexity: O(batch × out_channels × in_channels × kernel_h × kernel_w × out_h × out_w)
```

### 6. Backward Propagation

#### Dense Layer
```cpp
void DenseLayer::backprop(const Mat<T>& d_output) {
    // d_output: [batch, output_dim]

    // Gradient w.r.t. pre-activation
    Mat<T> dz = d_output ⊙ ActivateGrad(mActivated);

    // Gradient w.r.t. weights
    d_weights = Mult(mInput.T(), dz);  // [input_dim, output_dim]

    // Gradient w.r.t. bias
    d_bias = SumRows(dz);  // [output_dim]

    // Gradient w.r.t. input (for previous layer)
    d_input = Mult(dz, weights.T());  // [batch, input_dim]
}
```

#### Convolutional Layer
```cpp
void ConvLayer::backprop(const Tensor<T>& d_output) {
    // d_output: [batch, out_channels, out_h, out_w]

    // Gradient w.r.t. pre-activation
    Tensor<T> dz = d_output ⊙ ActivateGrad(mActivated);

    // Convert to column format
    Mat<T> d_col = tensor2col(dz);
    Mat<T> input_col = im2col(mInput, ...);

    // Gradient w.r.t. kernels
    d_kernels = Mult(d_col.T(), input_col);  // Reshape to kernel shape

    // Gradient w.r.t. bias
    d_bias = SumOverSpatialDims(dz);  // [out_channels]

    // Gradient w.r.t. input (for previous layer)
    // This is the transposed convolution (deconvolution)
    Mat<T> d_input_col = Mult(d_col, kernel_mat);
    Tensor<T> d_input = col2im(d_input_col, ...);
}
```

**Key Insight**: Backprop through convolution is itself a convolution (with transposed/rotated kernels).

---

## Changes Required for CNN Support

### 1. New Data Structures

#### Tensor Class
```cpp
template <typename T>
class Tensor {
public:
    std::vector<size_t> shape;        // [dim0, dim1, ..., dimN]
    std::shared_ptr<T[]> data;        // Contiguous memory

    // Construction
    Tensor(std::vector<size_t> shape);
    static Tensor<T> zeros(std::vector<size_t> shape);
    static Tensor<T> ones(std::vector<size_t> shape);

    // Indexing (variadic template)
    template<typename... Indices>
    T& operator()(Indices... indices);

    // Shape manipulation
    Tensor<T> reshape(std::vector<size_t> newShape);
    Tensor<T> transpose(std::vector<int> axes);
    Tensor<T> flatten();

    // Interop with Mat<T>
    static Tensor<T> fromMat(const Mat<T>& mat, std::vector<size_t> shape);
    Mat<T> toMat() const;

    // Element-wise operations
    Tensor<T> operator+(const Tensor<T>& other);
    Tensor<T> operator*(T scalar);
    // ... etc
};
```

**Rationale**:
- CNNs need 3D/4D data representation
- RNNs will need 3D (batch, time, features)
- Keeps `Mat<T>` for dense layers (no performance regression)

### 2. Convolution Utilities (im2col)

#### What is im2col?

**im2col** (image-to-column) transforms convolution into matrix multiplication:

```
Input image (1 channel, 4×4):
[1  2  3  4]
[5  6  7  8]
[9  10 11 12]
[13 14 15 16]

Kernel (3×3), Stride=1, Padding=0 → Output will be 2×2

im2col extracts all 3×3 patches:
Patch 1:  [1 2 3 5 6 7 9 10 11]   → Top-left
Patch 2:  [2 3 4 6 7 8 10 11 12]  → Top-right
Patch 3:  [5 6 7 9 10 11 13 14 15]→ Bottom-left
Patch 4:  [6 7 8 10 11 12 14 15 16]→ Bottom-right

Column matrix (4 patches, 9 values each):
[1  2  3  5  6  7  9  10 11]
[2  3  4  6  7  8  10 11 12]
[5  6  7  9  10 11 13 14 15]
[6  7  8  10 11 12 14 15 16]

Now convolution becomes:
Output = Column_Matrix × Kernel_Vector^T
```

**Implementation**:
```cpp
template <typename T>
Mat<T> im2col(
    const Tensor<T>& input,   // [batch, channels, height, width]
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    // Calculate output dimensions
    int out_h = (height + 2*pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2*pad_w - kernel_w) / stride_w + 1;

    // Create column matrix
    int col_rows = batch * out_h * out_w;
    int col_cols = channels * kernel_h * kernel_w;
    Mat<T> col(col_rows, col_cols);

    // Extract patches
    for (int b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int row_idx = (b * out_h * out_w) + (oh * out_w) + ow;

                // Extract kernel_h × kernel_w patch
                for (int c = 0; c < channels; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int h = oh * stride_h + kh - pad_h;
                            int w = ow * stride_w + kw - pad_w;

                            // Padding
                            T val = (h >= 0 && h < height && w >= 0 && w < width)
                                    ? input(b, c, h, w)
                                    : 0;

                            int col_idx = (c * kernel_h * kernel_w) + (kh * kernel_w) + kw;
                            col(row_idx, col_idx) = val;
                        }
                    }
                }
            }
        }
    }

    return col;
}
```

**col2im** (inverse operation):
```cpp
template <typename T>
Tensor<T> col2im(
    const Mat<T>& col,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    // Reverse the im2col process
    // Accumulate values that appear in multiple patches
    // ... implementation
}
```

### 3. Layer Type Hierarchy

```cpp
// Base interface (already exists)
template <typename T>
class ILayer {
public:
    virtual void feed(const Mat<T>& in, int epoch) = 0;
    virtual void backprop() = 0;
    virtual void init() = 0;
    virtual std::vector<Mat<T>>& getParameters() = 0;
    // ... other methods
};

// Dense layer (refactored from existing Layer<T>)
template <typename T>
class DenseLayer : public ILayer<T> {
private:
    Mat<T> mWeights;
    Mat<T> mBias;
    // ... existing implementation

public:
    void feed(const Mat<T>& in, int epoch) override;
    void backprop() override;
    // ...
};

// NEW: Convolutional layer
template <typename T>
class Conv2D : public ILayer<T> {
private:
    Tensor<T> mKernels;  // [out_channels, in_channels, kernel_h, kernel_w]
    Tensor<T> mBias;     // [out_channels]

    int mOutChannels, mInChannels;
    int mKernelH, mKernelW;
    int mStrideH, mStrideW;
    int mPadH, mPadW;

    // Cached for backprop
    Mat<T> mInputCol;
    Tensor<T> mInput;

public:
    Conv2D(int out_channels, int kernel_h, int kernel_w,
           ActivationType act = RELU,
           int stride = 1, int padding = 0);

    void feed(const Mat<T>& in, int epoch) override;
    void backprop() override;
    void init() override;
};

// NEW: Pooling layer
template <typename T>
class MaxPool2D : public ILayer<T> {
private:
    int mPoolH, mPoolW;
    int mStrideH, mStrideW;

    // For backprop: remember max indices
    std::vector<std::vector<int>> mMaxIndices;

public:
    MaxPool2D(int pool_size, int stride = -1);

    void feed(const Mat<T>& in, int epoch) override;
    void backprop() override;
    void init() override { } // No parameters
};
```

### 4. Network Modifications

#### Shape Tracking
```cpp
template <typename T>
class ILayer {
protected:
    std::vector<size_t> mInputShape;   // Expected input shape
    std::vector<size_t> mOutputShape;  // Produced output shape

public:
    void setInputShape(std::vector<size_t> shape) { mInputShape = shape; }
    std::vector<size_t> getOutputShape() const { return mOutputShape; }
};
```

#### Automatic Flattening
```cpp
// When connecting Conv layer → Dense layer
void Network<T>::connect(ILayer<T>* from, ILayer<T>* to) {
    // Check if we need to flatten
    bool from_is_spatial = (from->getOutputShape().size() > 1);
    bool to_is_dense = dynamic_cast<DenseLayer<T>*>(to) != nullptr;

    if (from_is_spatial && to_is_dense) {
        // Insert automatic flatten layer
        auto* flatten = new FlattenLayer<T>();
        connectLayers(from, flatten);
        connectLayers(flatten, to);
    } else {
        connectLayers(from, to);
    }
}
```

### 5. Parameter Ownership Change

**Current** (Network-owned):
```cpp
class Network {
    map<ILayer*, Mat<T>> weights;  // Network owns all weights
};
```

**New** (Layer-owned):
```cpp
class ILayer {
    virtual std::vector<Mat<T>>& getParameters() = 0;
    virtual std::vector<Mat<T>>& getGradients() = 0;
};

class DenseLayer : public ILayer {
    std::vector<Mat<T>> mParameters;  // {weights, bias}
    std::vector<Mat<T>> mGradients;   // Gradient buffers

    std::vector<Mat<T>>& getParameters() override {
        if (mParameters.empty()) {
            mParameters = {mWeights, mBias};
        }
        return mParameters;
    }
};

class Conv2D : public ILayer {
    std::vector<Mat<T>> mParameters;  // {kernels_as_mat, bias_as_mat}

    std::vector<Mat<T>>& getParameters() override {
        if (mParameters.empty()) {
            mParameters = {mKernels.toMat(), mBias.toMat()};
        }
        return mParameters;
    }
};
```

**Rationale**:
- Dense layers: weights are per-connection
- Conv layers: kernels are layer-internal (not per-connection)
- Each layer knows best how to store its parameters

---

## Pluggable Architecture Pattern

### Dependency Injection Principle

Different layer types are **injected** into the network as dependencies:

```cpp
// Create network container
Network<float>* net = new Network<float>();

// Inject different layer types
ILayer<float>* conv = new Conv2D<float>(32, 3, 3, RELU);
ILayer<float>* pool = new MaxPool2D<float>(2, 2);
ILayer<float>* dense = new DenseLayer<float>(10, SOFTMAX);

// Network treats all layers uniformly via ILayer interface
net->addLayer(conv);
net->addLayer(pool);
net->addLayer(dense);

// Polymorphic behavior - each layer handles its own forward/backward
net->train(data, labels);
```

### Benefits

1. **Open/Closed Principle**: Open for extension (add new layer types), closed for modification (Network class unchanged)

2. **Testability**: Each layer can be tested in isolation
   ```cpp
   TEST(Conv2DTest, ForwardPass) {
       Conv2D<float> layer(32, 3, 3, RELU);
       Tensor<float> input = Tensor<float>::random({1, 1, 28, 28});
       Tensor<float> output = layer.feed(input, 0);
       ASSERT_EQ(output.shape, {1, 32, 26, 26});
   }
   ```

3. **Composability**: Mix and match layer types
   ```cpp
   // All valid combinations:
   Dense → Dense → Dense          (Classic MLP)
   Conv → Pool → Conv → Pool → Dense  (CNN)
   Conv → Conv → Dense → Conv     (Mixed architecture)
   ```

4. **Extensibility**: Adding new layer types is straightforward
   ```cpp
   // Future: Add Batch Normalization
   class BatchNorm2D : public ILayer<T> {
       // Implement required interface
       void feed(...) override { /* normalize */ }
       void backprop() override { /* gradient */ }
   };

   // Use immediately
   net->addLayer(new Conv2D<float>(32, 3, 3));
   net->addLayer(new BatchNorm2D<float>());  // New layer type!
   net->addLayer(new MaxPool2D<float>(2, 2));
   ```

---

## Migration Path

### Backward Compatibility

**Existing code continues to work**:
```cpp
// Old code (still valid)
Layer<T>* layer = new Layer<T>(128, RELU);  // Layer is now alias to DenseLayer

// New code (explicit)
DenseLayer<T>* layer = new DenseLayer<T>(128, RELU);
```

**Type alias** in `network.h`:
```cpp
#ifndef NETWORK_NO_COMPAT
template <typename T>
using Layer = DenseLayer<T>;
#endif
```

### Adoption Strategy

**Phase 1**: Add CNN support (non-breaking)
- All existing tests pass unchanged
- New CNN layers available as opt-in

**Phase 2**: Update examples (gradual)
- Add CNN examples (MNIST, CIFAR-10)
- Keep existing MLP examples

**Phase 3**: Encourage migration (documentation)
- Update docs to recommend `DenseLayer` over `Layer`
- Provide migration guide

**Phase 4**: Deprecation (far future)
- Mark `Layer` alias as deprecated (compiler warning)
- Eventually remove alias in major version bump

---

## Performance Comparison

### Parameter Efficiency

**Task**: Classify 28×28 MNIST images

| Architecture | Parameters | Memory | Accuracy |
|--------------|------------|--------|----------|
| Dense (784→128→10) | 100,480 | ~400 KB | ~97% |
| CNN (Conv32→Pool→Dense10) | 3,210 | ~13 KB | ~99% |
| **Reduction** | **31× fewer** | **30× less** | **+2% better** |

### Training Speed

**On CPU (single thread)**:
- Dense: ~1000 samples/sec
- CNN (im2col): ~200 samples/sec (5× slower per sample)

**But CNN trains better**:
- Dense: 50 epochs to 97%
- CNN: 10 epochs to 99%
- **Wall-clock time**: CNN wins despite slower per-sample (fewer epochs needed)

### Memory Trade-offs

**im2col overhead**:
- Input: 28×28×1 = 784 values
- im2col buffer: ~2×input = 1,568 values (temporary)
- Trade-off: 2× memory for cleaner implementation

---

## Summary Table: ANN vs CNN

| Feature | Dense/ANN | Convolutional/CNN |
|---------|-----------|-------------------|
| **Best For** | Structured data, small inputs | Images, spatial data |
| **Operation** | Matrix multiplication | Convolution (via im2col) |
| **Connectivity** | Fully connected | Locally connected |
| **Parameters** | O(n_in × n_out) | O(k² × c_in × c_out) |
| **Translation** | Not invariant | Equivariant |
| **Spatial Structure** | Lost (flattened) | Preserved |
| **Data Type** | Mat<T> (2D) | Tensor<T> (3D/4D) |
| **Implementation** | ~100 lines | ~300 lines |
| **Backprop** | Standard chain rule | Transposed convolution |

---

## Next Steps

1. **Read** `ARCHITECTURE_DESIGN.md` for implementation details
2. **Review** test files: `test_conv_layer.cpp`, `test_cnn_network.cpp`
3. **Try** the CNN MNIST example
4. **Extend** with your own layer types

---

## References

- **Goodfellow et al.** - "Deep Learning" (2016), Chapter 9: Convolutional Networks
- **CS231n** - Stanford CNN course: http://cs231n.github.io/convolutional-networks/
- **im2col explanation**: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Status**: Ready for Review
