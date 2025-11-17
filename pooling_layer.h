#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "tensor.h"
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>

/**
 * Pooling Layers for CNNs
 *
 * Implements 2D pooling operations (max and average) for downsampling spatial dimensions.
 * Pooling layers have no learnable parameters - they only perform downsampling.
 *
 * Input shape:  [batch, channels, height, width]
 * Output shape: [batch, channels, out_height, out_width]
 *
 * Pooling preserves the number of channels.
 */

/**
 * Max Pooling 2D
 *
 * Takes the maximum value within each pooling window.
 * Commonly used after convolutional layers to:
 * - Reduce spatial dimensions
 * - Provide translation invariance
 * - Reduce computation for subsequent layers
 *
 * Example:
 *   MaxPool2D<float> pool(2, 2);  // 2x2 pooling window, stride=2
 *   auto output = pool.forward(input);
 */
template <typename T>
class MaxPool2D {
public:
    /**
     * Constructor
     *
     * @param pool_h  Pooling window height
     * @param pool_w  Pooling window width
     * @param stride_h Vertical stride (default: same as pool_h)
     * @param stride_w Horizontal stride (default: same as pool_w)
     */
    MaxPool2D(
        int pool_h, int pool_w,
        int stride_h = -1, int stride_w = -1
    )
        : mPoolH(pool_h)
        , mPoolW(pool_w)
        , mStrideH(stride_h > 0 ? stride_h : pool_h)
        , mStrideW(stride_w > 0 ? stride_w : pool_w)
    {
        if (pool_h <= 0 || pool_w <= 0) {
            throw std::invalid_argument("MaxPool2D: Pool size must be positive");
        }
        if (mStrideH <= 0 || mStrideW <= 0) {
            throw std::invalid_argument("MaxPool2D: Stride must be positive");
        }
    }

    /**
     * Convenience constructor for square pooling
     */
    MaxPool2D(int pool_size, int stride = -1)
        : MaxPool2D(pool_size, pool_size, stride, stride)
    {}

    /**
     * Forward pass
     *
     * @param input Input tensor [batch, channels, height, width]
     * @return Output tensor [batch, channels, out_height, out_width]
     */
    Tensor<T> forward(const Tensor<T>& input) {
        if (input.ndim() != 4) {
            throw std::runtime_error("MaxPool2D: Input must be 4D [batch, channels, height, width]");
        }

        int batch = input.shape(0);
        int channels = input.shape(1);
        int in_h = input.shape(2);
        int in_w = input.shape(3);

        // Calculate output dimensions
        int out_h = (in_h - mPoolH) / mStrideH + 1;
        int out_w = (in_w - mPoolW) / mStrideW + 1;

        if (out_h <= 0 || out_w <= 0) {
            throw std::runtime_error("MaxPool2D: Invalid output dimensions");
        }

        // Cache input for backward pass
        mInput = input.copy();
        mBatchSize = batch;
        mChannels = channels;
        mInputH = in_h;
        mInputW = in_w;
        mOutputH = out_h;
        mOutputW = out_w;

        // Create output tensor
        Tensor<T> output({
            (size_t)batch,
            (size_t)channels,
            (size_t)out_h,
            (size_t)out_w
        });

        // Store max indices for backward pass
        mMaxIndices.clear();
        mMaxIndices.resize(batch * channels * out_h * out_w);

        // Perform max pooling
        size_t idx = 0;
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        // Find max in pooling window
                        T max_val = -std::numeric_limits<T>::infinity();
                        int max_h = 0, max_w = 0;

                        for (int ph = 0; ph < mPoolH; ++ph) {
                            for (int pw = 0; pw < mPoolW; ++pw) {
                                int h = oh * mStrideH + ph;
                                int w = ow * mStrideW + pw;

                                if (h < in_h && w < in_w) {
                                    T val = input(b, c, h, w);
                                    if (val > max_val) {
                                        max_val = val;
                                        max_h = h;
                                        max_w = w;
                                    }
                                }
                            }
                        }

                        output(b, c, oh, ow) = max_val;

                        // Store max index for backward pass
                        mMaxIndices[idx++] = {b, c, max_h, max_w};
                    }
                }
            }
        }

        return output;
    }

    /**
     * Backward pass
     *
     * Routes gradients only to the locations that had the max values during forward pass.
     *
     * @param d_output Gradient w.r.t. output [batch, channels, out_height, out_width]
     * @return Gradient w.r.t. input [batch, channels, in_height, in_width]
     */
    Tensor<T> backward(const Tensor<T>& d_output) {
        if (d_output.ndim() != 4) {
            throw std::runtime_error("MaxPool2D: d_output must be 4D");
        }

        // Initialize gradient tensor (all zeros)
        Tensor<T> d_input({
            (size_t)mBatchSize,
            (size_t)mChannels,
            (size_t)mInputH,
            (size_t)mInputW
        }, T(0));

        // Route gradients to max positions
        size_t idx = 0;
        for (int b = 0; b < mBatchSize; ++b) {
            for (int c = 0; c < mChannels; ++c) {
                for (int oh = 0; oh < mOutputH; ++oh) {
                    for (int ow = 0; ow < mOutputW; ++ow) {
                        auto& max_idx = mMaxIndices[idx++];
                        d_input(max_idx[0], max_idx[1], max_idx[2], max_idx[3]) +=
                            d_output(b, c, oh, ow);
                    }
                }
            }
        }

        return d_input;
    }

    // Getters
    int getOutputHeight() const { return mOutputH; }
    int getOutputWidth() const { return mOutputW; }

private:
    int mPoolH, mPoolW;
    int mStrideH, mStrideW;

    // Cached for backward pass
    Tensor<T> mInput;
    std::vector<std::array<int, 4>> mMaxIndices;  // [batch, channel, height, width]

    // Cached dimensions
    int mBatchSize, mChannels;
    int mInputH, mInputW;
    int mOutputH, mOutputW;
};

/**
 * Average Pooling 2D
 *
 * Takes the average value within each pooling window.
 * Less commonly used than max pooling, but sometimes preferred for:
 * - Smoother downsampling
 * - Global average pooling (GAP) before final classification
 *
 * Example:
 *   AvgPool2D<float> pool(2, 2);  // 2x2 pooling window
 *   auto output = pool.forward(input);
 */
template <typename T>
class AvgPool2D {
public:
    /**
     * Constructor
     *
     * @param pool_h  Pooling window height
     * @param pool_w  Pooling window width
     * @param stride_h Vertical stride (default: same as pool_h)
     * @param stride_w Horizontal stride (default: same as pool_w)
     */
    AvgPool2D(
        int pool_h, int pool_w,
        int stride_h = -1, int stride_w = -1
    )
        : mPoolH(pool_h)
        , mPoolW(pool_w)
        , mStrideH(stride_h > 0 ? stride_h : pool_h)
        , mStrideW(stride_w > 0 ? stride_w : pool_w)
    {
        if (pool_h <= 0 || pool_w <= 0) {
            throw std::invalid_argument("AvgPool2D: Pool size must be positive");
        }
        if (mStrideH <= 0 || mStrideW <= 0) {
            throw std::invalid_argument("AvgPool2D: Stride must be positive");
        }
    }

    /**
     * Convenience constructor for square pooling
     */
    AvgPool2D(int pool_size, int stride = -1)
        : AvgPool2D(pool_size, pool_size, stride, stride)
    {}

    /**
     * Forward pass
     *
     * @param input Input tensor [batch, channels, height, width]
     * @return Output tensor [batch, channels, out_height, out_width]
     */
    Tensor<T> forward(const Tensor<T>& input) {
        if (input.ndim() != 4) {
            throw std::runtime_error("AvgPool2D: Input must be 4D [batch, channels, height, width]");
        }

        int batch = input.shape(0);
        int channels = input.shape(1);
        int in_h = input.shape(2);
        int in_w = input.shape(3);

        // Calculate output dimensions
        int out_h = (in_h - mPoolH) / mStrideH + 1;
        int out_w = (in_w - mPoolW) / mStrideW + 1;

        if (out_h <= 0 || out_w <= 0) {
            throw std::runtime_error("AvgPool2D: Invalid output dimensions");
        }

        // Cache dimensions for backward pass
        mBatchSize = batch;
        mChannels = channels;
        mInputH = in_h;
        mInputW = in_w;
        mOutputH = out_h;
        mOutputW = out_w;

        // Create output tensor
        Tensor<T> output({
            (size_t)batch,
            (size_t)channels,
            (size_t)out_h,
            (size_t)out_w
        });

        // Perform average pooling
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        // Calculate average in pooling window
                        T sum = 0;
                        int count = 0;

                        for (int ph = 0; ph < mPoolH; ++ph) {
                            for (int pw = 0; pw < mPoolW; ++pw) {
                                int h = oh * mStrideH + ph;
                                int w = ow * mStrideW + pw;

                                if (h < in_h && w < in_w) {
                                    sum += input(b, c, h, w);
                                    count++;
                                }
                            }
                        }

                        output(b, c, oh, ow) = sum / T(count);
                    }
                }
            }
        }

        return output;
    }

    /**
     * Backward pass
     *
     * Distributes gradients evenly across the pooling window.
     *
     * @param d_output Gradient w.r.t. output [batch, channels, out_height, out_width]
     * @return Gradient w.r.t. input [batch, channels, in_height, in_width]
     */
    Tensor<T> backward(const Tensor<T>& d_output) {
        if (d_output.ndim() != 4) {
            throw std::runtime_error("AvgPool2D: d_output must be 4D");
        }

        // Initialize gradient tensor (all zeros)
        Tensor<T> d_input({
            (size_t)mBatchSize,
            (size_t)mChannels,
            (size_t)mInputH,
            (size_t)mInputW
        }, T(0));

        // Distribute gradients evenly across pooling windows
        for (int b = 0; b < mBatchSize; ++b) {
            for (int c = 0; c < mChannels; ++c) {
                for (int oh = 0; oh < mOutputH; ++oh) {
                    for (int ow = 0; ow < mOutputW; ++ow) {
                        // Count valid positions in pooling window
                        int count = 0;
                        for (int ph = 0; ph < mPoolH; ++ph) {
                            for (int pw = 0; pw < mPoolW; ++pw) {
                                int h = oh * mStrideH + ph;
                                int w = ow * mStrideW + pw;
                                if (h < mInputH && w < mInputW) {
                                    count++;
                                }
                            }
                        }

                        // Distribute gradient evenly
                        T grad_per_element = d_output(b, c, oh, ow) / T(count);

                        for (int ph = 0; ph < mPoolH; ++ph) {
                            for (int pw = 0; pw < mPoolW; ++pw) {
                                int h = oh * mStrideH + ph;
                                int w = ow * mStrideW + pw;

                                if (h < mInputH && w < mInputW) {
                                    d_input(b, c, h, w) += grad_per_element;
                                }
                            }
                        }
                    }
                }
            }
        }

        return d_input;
    }

    // Getters
    int getOutputHeight() const { return mOutputH; }
    int getOutputWidth() const { return mOutputW; }

private:
    int mPoolH, mPoolW;
    int mStrideH, mStrideW;

    // Cached dimensions
    int mBatchSize, mChannels;
    int mInputH, mInputW;
    int mOutputH, mOutputW;
};

/**
 * Global Average Pooling 2D
 *
 * Pools over the entire spatial dimensions, reducing to [batch, channels, 1, 1].
 * Commonly used as an alternative to fully-connected layers before classification.
 *
 * Example:
 *   GlobalAvgPool2D<float> gap;
 *   auto output = gap.forward(input);  // [batch, channels, h, w] → [batch, channels]
 */
template <typename T>
class GlobalAvgPool2D {
public:
    GlobalAvgPool2D() {}

    /**
     * Forward pass
     */
    Tensor<T> forward(const Tensor<T>& input) {
        if (input.ndim() != 4) {
            throw std::runtime_error("GlobalAvgPool2D: Input must be 4D");
        }

        int batch = input.shape(0);
        int channels = input.shape(1);
        int height = input.shape(2);
        int width = input.shape(3);

        // Cache for backward
        mInputH = height;
        mInputW = width;

        // Output: [batch, channels, 1, 1]
        Tensor<T> output({(size_t)batch, (size_t)channels, 1, 1});

        // Average over spatial dimensions
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                T sum = 0;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        sum += input(b, c, h, w);
                    }
                }
                output(b, c, 0, 0) = sum / T(height * width);
            }
        }

        return output;
    }

    /**
     * Backward pass
     */
    Tensor<T> backward(const Tensor<T>& d_output, const std::vector<size_t>& input_shape) {
        int batch = input_shape[0];
        int channels = input_shape[1];
        int height = input_shape[2];
        int width = input_shape[3];

        Tensor<T> d_input(input_shape, T(0));

        // Distribute gradient evenly
        T factor = T(1) / T(height * width);

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channels; ++c) {
                T grad = d_output(b, c, 0, 0) * factor;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        d_input(b, c, h, w) = grad;
                    }
                }
            }
        }

        return d_input;
    }

private:
    int mInputH, mInputW;
};

#endif // POOLING_LAYER_H
