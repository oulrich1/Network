#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "tensor.h"
#include "im2col.h"
#include "activation.h"
#include "Matrix/matrix.h"
#include <random>
#include <cmath>
#include <iostream>

// Use ml namespace types
using ml::ActivationType;

/**
 * Convolutional Layer (Conv2D)
 *
 * Implements 2D convolution operation with learnable kernels/filters.
 * Uses im2col for efficient computation via matrix multiplication.
 *
 * Input shape:  [batch, in_channels, height, width]
 * Output shape: [batch, out_channels, out_height, out_width]
 *
 * Example:
 *   Conv2D<float> conv(32, 3, 3, RELU);  // 32 filters, 3x3 kernel
 *   conv.setInputChannels(1);            // Grayscale input
 *   conv.init();
 *   auto output = conv.forward(input);
 */
template <typename T>
class Conv2D {
public:
    /**
     * Constructor
     *
     * @param out_channels  Number of output channels (filters)
     * @param kernel_h      Kernel height
     * @param kernel_w      Kernel width
     * @param activation    Activation function type
     * @param stride_h      Vertical stride (default: 1)
     * @param stride_w      Horizontal stride (default: 1)
     * @param pad_h         Vertical padding (default: 0)
     * @param pad_w         Horizontal padding (default: 0)
     */
    Conv2D(
        int out_channels,
        int kernel_h, int kernel_w,
        ActivationType activation = ActivationType::RELU,
        int stride_h = 1, int stride_w = 1,
        int pad_h = 0, int pad_w = 0
    )
        : mOutChannels(out_channels)
        , mKernelH(kernel_h)
        , mKernelW(kernel_w)
        , mStrideH(stride_h)
        , mStrideW(stride_w)
        , mPadH(pad_h)
        , mPadW(pad_w)
        , mActivationType(activation)
        , mActivationAlpha(0.01)  // For LeakyReLU/ELU
        , mInChannels(0)
        , mInitialized(false)
    {
        if (out_channels <= 0 || kernel_h <= 0 || kernel_w <= 0) {
            throw std::invalid_argument("Conv2D: Invalid parameters");
        }
        if (stride_h <= 0 || stride_w <= 0) {
            throw std::invalid_argument("Conv2D: Stride must be positive");
        }
        if (pad_h < 0 || pad_w < 0) {
            throw std::invalid_argument("Conv2D: Padding must be non-negative");
        }
    }

    /**
     * Set number of input channels (must be called before init)
     */
    void setInputChannels(int in_channels) {
        if (mInitialized) {
            throw std::runtime_error("Conv2D: Cannot change input channels after initialization");
        }
        if (in_channels <= 0) {
            throw std::invalid_argument("Conv2D: Input channels must be positive");
        }
        mInChannels = in_channels;
    }

    /**
     * Initialize weights and biases
     * Uses He initialization for ReLU, Xavier for others
     */
    void init() {
        if (mInChannels == 0) {
            throw std::runtime_error("Conv2D: Must set input channels before initialization");
        }

        // Initialize kernels: [out_channels, in_channels, kernel_h, kernel_w]
        size_t kernel_size = mOutChannels * mInChannels * mKernelH * mKernelW;
        mKernels = Tensor<T>({
            (size_t)mOutChannels,
            (size_t)mInChannels,
            (size_t)mKernelH,
            (size_t)mKernelW
        });

        // Initialize bias: [out_channels]
        mBias = Tensor<T>({(size_t)mOutChannels}, T(0));

        // Weight initialization
        std::random_device rd;
        std::mt19937 gen(rd());

        // He initialization for ReLU: std = sqrt(2 / fan_in)
        // Xavier for others: std = sqrt(2 / (fan_in + fan_out))
        int fan_in = mInChannels * mKernelH * mKernelW;
        int fan_out = mOutChannels * mKernelH * mKernelW;

        T stddev;
        if (mActivationType == ActivationType::RELU || mActivationType == ActivationType::LEAKY_RELU) {
            stddev = std::sqrt(T(2.0) / T(fan_in));  // He initialization
        } else {
            stddev = std::sqrt(T(2.0) / T(fan_in + fan_out));  // Xavier
        }

        std::normal_distribution<T> dist(T(0), stddev);

        // Initialize kernel weights
        for (size_t i = 0; i < kernel_size; ++i) {
            mKernels(i) = dist(gen);
        }

        // Bias typically initialized to small positive value for ReLU
        if (mActivationType == ActivationType::RELU || mActivationType == ActivationType::LEAKY_RELU) {
            for (int i = 0; i < mOutChannels; ++i) {
                mBias(i) = T(0.01);
            }
        }

        // Initialize gradient accumulators
        mKernelGrad = Tensor<T>::zeros({
            (size_t)mOutChannels,
            (size_t)mInChannels,
            (size_t)mKernelH,
            (size_t)mKernelW
        });
        mBiasGrad = Tensor<T>::zeros({(size_t)mOutChannels});

        mInitialized = true;
    }

    /**
     * Forward pass
     *
     * @param input  Input tensor [batch, in_channels, height, width]
     * @return Output tensor [batch, out_channels, out_height, out_width]
     */
    Tensor<T> forward(const Tensor<T>& input) {
        if (!mInitialized) {
            throw std::runtime_error("Conv2D: Layer not initialized. Call init() first.");
        }

        if (input.ndim() != 4) {
            throw std::runtime_error("Conv2D: Input must be 4D [batch, channels, height, width]");
        }

        int batch = input.shape(0);
        int in_channels = input.shape(1);
        int in_h = input.shape(2);
        int in_w = input.shape(3);

        if (in_channels != mInChannels) {
            throw std::runtime_error("Conv2D: Input channels mismatch");
        }

        // Calculate output dimensions
        int out_h, out_w;
        nn::im2col_get_output_dims<T>(
            in_h, in_w,
            mKernelH, mKernelW,
            mStrideH, mStrideW,
            mPadH, mPadW,
            out_h, out_w
        );

        // Cache input for backward pass
        mInput = input.copy();
        mBatchSize = batch;
        mInputH = in_h;
        mInputW = in_w;
        mOutputH = out_h;
        mOutputW = out_w;

        // Step 1: im2col - transform input into column matrix
        mInputCol = nn::im2col<T>(
            input,
            mKernelH, mKernelW,
            mStrideH, mStrideW,
            mPadH, mPadW
        );
        // mInputCol: [batch * out_h * out_w, in_channels * kernel_h * kernel_w]

        // Step 2: Reshape kernels to 2D matrix
        // From: [out_channels, in_channels, kernel_h, kernel_w]
        // To: [out_channels, in_channels * kernel_h * kernel_w]
        int kernel_flat_size = mInChannels * mKernelH * mKernelW;
        ml::Mat<T> kernel_mat(mOutChannels, kernel_flat_size);

        for (int oc = 0; oc < mOutChannels; ++oc) {
            int idx = 0;
            for (int ic = 0; ic < mInChannels; ++ic) {
                for (int kh = 0; kh < mKernelH; ++kh) {
                    for (int kw = 0; kw < mKernelW; ++kw) {
                        kernel_mat.setAt(oc, idx++, mKernels(oc, ic, kh, kw));
                    }
                }
            }
        }

        // Step 3: Matrix multiplication
        // mInputCol: [N, K] where N = batch * out_h * out_w, K = in_ch * kh * kw
        // kernel_mat^T: [K, M] where M = out_channels
        // Result: [N, M]
        auto kernel_mat_T = kernel_mat.Copy();
        kernel_mat_T.Transpose();  // Now [kernel_flat_size, out_channels]

        ml::Mat<T> output_mat = mInputCol.Mult(kernel_mat_T);
        // output_mat: [batch * out_h * out_w, out_channels]

        // Step 4: Add bias
        auto output_size = output_mat.size();
        for (int row = 0; row < output_size.cy; ++row) {
            for (int col = 0; col < output_size.cx; ++col) {
                T val = output_mat.getAt(row, col) + mBias(col);
                output_mat.setAt(row, col, val);
            }
        }

        // Step 5: Reshape to 4D tensor
        Tensor<T> output_4d({
            (size_t)batch,
            (size_t)mOutChannels,
            (size_t)out_h,
            (size_t)out_w
        });

        int row = 0;
        for (int b = 0; b < batch; ++b) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    for (int oc = 0; oc < mOutChannels; ++oc) {
                        output_4d(b, oc, oh, ow) = output_mat.getAt(row, oc);
                    }
                    row++;
                }
            }
        }

        // Step 6: Apply activation (cache pre-activation for backward)
        mPreActivation = output_4d.copy();
        mOutput = applyActivation(output_4d);

        return mOutput;
    }

    /**
     * Backward pass
     *
     * @param d_output  Gradient w.r.t. output [batch, out_channels, out_height, out_width]
     * @return Gradient w.r.t. input [batch, in_channels, in_height, in_width]
     */
    Tensor<T> backward(const Tensor<T>& d_output) {
        if (!mInitialized) {
            throw std::runtime_error("Conv2D: Cannot backprop on uninitialized layer");
        }

        // Step 1: Gradient through activation
        Tensor<T> d_preact = d_output * computeActivationGradient(mPreActivation);

        // Step 2: Reshape gradient to 2D matrix
        // From: [batch, out_channels, out_h, out_w]
        // To: [batch * out_h * out_w, out_channels]
        int N = mBatchSize * mOutputH * mOutputW;
        ml::Mat<T> d_output_mat(N, mOutChannels);

        int row = 0;
        for (int b = 0; b < mBatchSize; ++b) {
            for (int oh = 0; oh < mOutputH; ++oh) {
                for (int ow = 0; ow < mOutputW; ++ow) {
                    for (int oc = 0; oc < mOutChannels; ++oc) {
                        d_output_mat.setAt(row, oc, d_preact(b, oc, oh, ow));
                    }
                    row++;
                }
            }
        }

        // Step 3: Gradient w.r.t. bias (sum over spatial dimensions)
        mBiasGrad.fill(0);
        auto d_out_size = d_output_mat.size();
        for (int row = 0; row < d_out_size.cy; ++row) {
            for (int col = 0; col < d_out_size.cx; ++col) {
                mBiasGrad(col) += d_output_mat.getAt(row, col);
            }
        }

        // Step 4: Gradient w.r.t. kernels
        // d_kernel = d_output_mat^T * mInputCol
        // d_output_mat: [N, out_channels]
        // mInputCol: [N, in_channels * kh * kw]
        // Result: [out_channels, in_channels * kh * kw]
        auto d_output_mat_T = d_output_mat.Copy();
        d_output_mat_T.Transpose();  // [out_channels, N]

        ml::Mat<T> d_kernel_flat = d_output_mat_T.Mult(mInputCol);
        // [out_channels, in_channels * kh * kw]

        // Reshape to kernel shape and accumulate gradients
        mKernelGrad.fill(0);
        int kernel_flat_size = mInChannels * mKernelH * mKernelW;
        for (int oc = 0; oc < mOutChannels; ++oc) {
            int idx = 0;
            for (int ic = 0; ic < mInChannels; ++ic) {
                for (int kh = 0; kh < mKernelH; ++kh) {
                    for (int kw = 0; kw < mKernelW; ++kw) {
                        mKernelGrad(oc, ic, kh, kw) = d_kernel_flat.getAt(oc, idx++);
                    }
                }
            }
        }

        // Step 5: Gradient w.r.t. input
        // d_input_col = d_output_mat * kernel_mat
        // d_output_mat: [N, out_channels]
        // kernel_mat: [out_channels, in_channels * kh * kw]
        // Result: [N, in_channels * kh * kw]

        // Reshape kernels to 2D matrix
        ml::Mat<T> kernel_mat(mOutChannels, kernel_flat_size);
        for (int oc = 0; oc < mOutChannels; ++oc) {
            int idx = 0;
            for (int ic = 0; ic < mInChannels; ++ic) {
                for (int kh = 0; kh < mKernelH; ++kh) {
                    for (int kw = 0; kw < mKernelW; ++kw) {
                        kernel_mat.setAt(oc, idx++, mKernels(oc, ic, kh, kw));
                    }
                }
            }
        }

        ml::Mat<T> d_input_col = d_output_mat.Mult(kernel_mat);

        // Step 6: col2im to get gradient w.r.t. input
        Tensor<T> d_input = nn::col2im<T>(
            d_input_col,
            mBatchSize, mInChannels, mInputH, mInputW,
            mKernelH, mKernelW,
            mStrideH, mStrideW,
            mPadH, mPadW
        );

        return d_input;
    }

    /**
     * Update weights using gradients
     *
     * @param learning_rate  Learning rate
     */
    void updateWeights(T learning_rate) {
        // Simple SGD update
        for (size_t i = 0; i < mKernels.size(); ++i) {
            mKernels(i) -= learning_rate * mKernelGrad(i);
        }

        for (size_t i = 0; i < mBias.size(); ++i) {
            mBias(i) -= learning_rate * mBiasGrad(i);
        }
    }

    // Getters
    const Tensor<T>& getKernels() const { return mKernels; }
    const Tensor<T>& getBias() const { return mBias; }
    const Tensor<T>& getKernelGradients() const { return mKernelGrad; }
    const Tensor<T>& getBiasGradients() const { return mBiasGrad; }
    int getOutputHeight() const { return mOutputH; }
    int getOutputWidth() const { return mOutputW; }
    int getOutputChannels() const { return mOutChannels; }

    // Setters for testing
    void setKernels(const Tensor<T>& kernels) { mKernels = kernels; }
    void setBias(const Tensor<T>& bias) { mBias = bias; }

private:
    /**
     * Apply activation function
     */
    Tensor<T> applyActivation(const Tensor<T>& input) {
        Tensor<T> output = input.copy();

        for (size_t i = 0; i < input.size(); ++i) {
            T val = input(i);
            output(i) = activateScalar(val, mActivationType, mActivationAlpha);
        }

        return output;
    }

    /**
     * Compute activation gradient (element-wise)
     */
    Tensor<T> computeActivationGradient(const Tensor<T>& preactivation) {
        Tensor<T> grad({preactivation.shape()});

        for (size_t i = 0; i < preactivation.size(); ++i) {
            T val = preactivation(i);
            grad(i) = activateGradScalar(val, mActivationType, mActivationAlpha);
        }

        return grad;
    }

    /**
     * Scalar activation (helper)
     */
    T activateScalar(T x, ActivationType type, T alpha) {
        switch (type) {
            case ActivationType::SIGMOID: return T(1) / (T(1) + std::exp(-x));
            case ActivationType::TANH: return std::tanh(x);
            case ActivationType::RELU: return std::max(T(0), x);
            case ActivationType::LEAKY_RELU: return x > 0 ? x : alpha * x;
            case ActivationType::ELU: return x > 0 ? x : alpha * (std::exp(x) - T(1));
            case ActivationType::LINEAR: return x;
            default: return x;
        }
    }

    /**
     * Scalar activation gradient (helper)
     */
    T activateGradScalar(T x, ActivationType type, T alpha) {
        switch (type) {
            case ActivationType::SIGMOID: {
                T s = T(1) / (T(1) + std::exp(-x));
                return s * (T(1) - s);
            }
            case ActivationType::TANH: {
                T t = std::tanh(x);
                return T(1) - t * t;
            }
            case ActivationType::RELU: return x > 0 ? T(1) : T(0);
            case ActivationType::LEAKY_RELU: return x > 0 ? T(1) : alpha;
            case ActivationType::ELU: return x > 0 ? T(1) : alpha * std::exp(x);
            case ActivationType::LINEAR: return T(1);
            default: return T(1);
        }
    }

private:
    // Layer configuration
    int mOutChannels;
    int mInChannels;
    int mKernelH, mKernelW;
    int mStrideH, mStrideW;
    int mPadH, mPadW;
    ActivationType mActivationType;
    T mActivationAlpha;

    // Learnable parameters
    Tensor<T> mKernels;      // [out_channels, in_channels, kernel_h, kernel_w]
    Tensor<T> mBias;         // [out_channels]

    // Gradients
    Tensor<T> mKernelGrad;
    Tensor<T> mBiasGrad;

    // Cached values for backward pass
    Tensor<T> mInput;        // Input tensor
    ml::Mat<T> mInputCol;    // im2col'd input
    Tensor<T> mPreActivation; // Before activation
    Tensor<T> mOutput;       // After activation

    // Cached dimensions
    int mBatchSize;
    int mInputH, mInputW;
    int mOutputH, mOutputW;

    // State
    bool mInitialized;
};

#endif // CONV_LAYER_H
