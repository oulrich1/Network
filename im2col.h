#ifndef IM2COL_H
#define IM2COL_H

#include "tensor.h"
#include "Matrix/matrix.h"
#include <cstring>

/**
 * im2col (image-to-column) and col2im (column-to-image) utilities
 *
 * These functions transform convolution operations into matrix multiplications,
 * which is the standard approach used by modern deep learning frameworks
 * (Caffe, PyTorch, TensorFlow) for CPU-based convolutions.
 *
 * Reference: "High Performance Convolutional Neural Networks for Document Processing"
 *            by Chellapilla et al. (2006)
 */

namespace nn {

/**
 * im2col: Transform image patches into columns for convolution via matrix multiplication
 *
 * Extracts all spatial patches from the input tensor and arranges them as columns
 * in a 2D matrix. This allows convolution to be computed as matrix multiplication.
 *
 * Input shape: [batch, channels, height, width]
 * Output shape: [batch * out_h * out_w, channels * kernel_h * kernel_w]
 *
 * Example:
 *   Input: [1, 1, 4, 4] image
 *   Kernel: 3x3, stride=1, padding=0
 *   Output: [1 * 2 * 2, 1 * 3 * 3] = [4, 9] matrix
 *   Each row contains one 3x3 patch from the image
 *
 * @param input      Input tensor [batch, channels, height, width]
 * @param kernel_h   Kernel height
 * @param kernel_w   Kernel width
 * @param stride_h   Vertical stride
 * @param stride_w   Horizontal stride
 * @param pad_h      Vertical padding
 * @param pad_w      Horizontal padding
 * @return Column matrix [batch * out_h * out_w, channels * kernel_h * kernel_w]
 */
template <typename T>
ml::Mat<T> im2col(
    const Tensor<T>& input,
    int kernel_h, int kernel_w,
    int stride_h = 1, int stride_w = 1,
    int pad_h = 0, int pad_w = 0
) {
    // Input dimensions
    if (input.ndim() != 4) {
        throw std::runtime_error("im2col: Input must be 4D tensor [batch, channels, height, width]");
    }

    int batch = input.shape(0);
    int channels = input.shape(1);
    int height = input.shape(2);
    int width = input.shape(3);

    // Output dimensions
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error("im2col: Invalid output dimensions (check kernel size, stride, padding)");
    }

    // Create column matrix
    int col_rows = batch * out_h * out_w;
    int col_cols = channels * kernel_h * kernel_w;
    ml::Mat<T> col_mat(col_rows, col_cols, 0);

    // Extract patches
    int row_idx = 0;
    for (int b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                // For this output position, extract the patch
                int col_idx = 0;

                for (int c = 0; c < channels; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            // Calculate input position
                            int h = oh * stride_h + kh - pad_h;
                            int w = ow * stride_w + kw - pad_w;

                            // Apply padding (zero if out of bounds)
                            T val = 0;
                            if (h >= 0 && h < height && w >= 0 && w < width) {
                                val = input(b, c, h, w);
                            }

                            col_mat.setAt(row_idx, col_idx, val);
                            col_idx++;
                        }
                    }
                }

                row_idx++;
            }
        }
    }

    return col_mat;
}

/**
 * col2im: Inverse of im2col
 *
 * Accumulates columns back into image format. This is used during backpropagation
 * to compute gradients with respect to the input.
 *
 * Note: This is an accumulation operation (not assignment) because patches overlap
 * with non-unit strides, so gradients from multiple output positions contribute
 * to the same input position.
 *
 * @param col_mat    Column matrix [batch * out_h * out_w, channels * kernel_h * kernel_w]
 * @param batch      Batch size
 * @param channels   Number of input channels
 * @param height     Input height
 * @param width      Input width
 * @param kernel_h   Kernel height
 * @param kernel_w   Kernel width
 * @param stride_h   Vertical stride
 * @param stride_w   Horizontal stride
 * @param pad_h      Vertical padding
 * @param pad_w      Horizontal padding
 * @return Reconstructed tensor [batch, channels, height, width]
 */
template <typename T>
Tensor<T> col2im(
    const ml::Mat<T>& col_mat,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int stride_h = 1, int stride_w = 1,
    int pad_h = 0, int pad_w = 0
) {
    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Verify column matrix dimensions
    auto col_size = col_mat.size();
    int expected_rows = batch * out_h * out_w;
    int expected_cols = channels * kernel_h * kernel_w;

    if (col_size.cy != expected_rows || col_size.cx != expected_cols) {
        throw std::runtime_error("col2im: Column matrix dimensions do not match expected shape");
    }

    // Create output tensor (initialized to zero for accumulation)
    Tensor<T> output({(size_t)batch, (size_t)channels, (size_t)height, (size_t)width}, T(0));

    // Accumulate columns back into image
    int row_idx = 0;
    for (int b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int col_idx = 0;

                for (int c = 0; c < channels; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            // Calculate input position
                            int h = oh * stride_h + kh - pad_h;
                            int w = ow * stride_w + kw - pad_w;

                            // Accumulate if within bounds
                            if (h >= 0 && h < height && w >= 0 && w < width) {
                                T val = col_mat.getAt(row_idx, col_idx);
                                output(b, c, h, w) += val;
                            }

                            col_idx++;
                        }
                    }
                }

                row_idx++;
            }
        }
    }

    return output;
}

/**
 * im2col_get_output_dims: Calculate output dimensions for convolution
 *
 * Helper function to compute output spatial dimensions given input size and conv params.
 *
 * @param input_h    Input height
 * @param input_w    Input width
 * @param kernel_h   Kernel height
 * @param kernel_w   Kernel width
 * @param stride_h   Vertical stride
 * @param stride_w   Horizontal stride
 * @param pad_h      Vertical padding
 * @param pad_w      Horizontal padding
 * @param out_h      [OUT] Output height
 * @param out_w      [OUT] Output width
 */
template <typename T>
void im2col_get_output_dims(
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int& out_h, int& out_w
) {
    out_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    out_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error(
            "Invalid convolution parameters: output dimensions are non-positive. "
            "Check kernel size, stride, and padding."
        );
    }
}

/**
 * im2col_backward_data: Compute gradient with respect to input
 *
 * This is essentially col2im, but kept as separate function for clarity.
 * During backpropagation, gradients flow backward through the im2col operation
 * by accumulating them via col2im.
 *
 * @param d_output_col   Gradient w.r.t. column matrix
 * @param batch          Batch size
 * @param channels       Number of input channels
 * @param height         Input height
 * @param width          Input width
 * @param kernel_h       Kernel height
 * @param kernel_w       Kernel width
 * @param stride_h       Vertical stride
 * @param stride_w       Horizontal stride
 * @param pad_h          Vertical padding
 * @param pad_w          Horizontal padding
 * @return Gradient w.r.t. input [batch, channels, height, width]
 */
template <typename T>
Tensor<T> im2col_backward_data(
    const ml::Mat<T>& d_output_col,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int stride_h = 1, int stride_w = 1,
    int pad_h = 0, int pad_w = 0
) {
    return col2im<T>(
        d_output_col,
        batch, channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );
}

/**
 * im2col_backward_kernel: Compute gradient with respect to kernel
 *
 * Computes ∂L/∂W where W is the kernel/filter weights.
 * This is done by matrix multiplication: d_kernel = input_col^T * d_output_col
 *
 * @param input_col      im2col'd input [batch*out_h*out_w, channels*kernel_h*kernel_w]
 * @param d_output_col   Gradient w.r.t. output [batch*out_h*out_w, out_channels]
 * @param out_channels   Number of output channels (filters)
 * @param in_channels    Number of input channels
 * @param kernel_h       Kernel height
 * @param kernel_w       Kernel width
 * @return Gradient w.r.t. kernel [out_channels, in_channels * kernel_h * kernel_w]
 */
template <typename T>
ml::Mat<T> im2col_backward_kernel(
    const ml::Mat<T>& input_col,
    const ml::Mat<T>& d_output_col,
    int out_channels, int in_channels,
    int kernel_h, int kernel_w
) {
    // d_kernel = d_output_col^T * input_col
    // d_output_col: [batch*out_h*out_w, out_channels]
    // input_col: [batch*out_h*out_w, in_channels*kernel_h*kernel_w]
    // Result: [out_channels, in_channels*kernel_h*kernel_w]

    auto d_output_T = d_output_col.Copy();
    d_output_T.Transpose();  // Now [out_channels, batch*out_h*out_w]

    ml::Mat<T> d_kernel = d_output_T.Mult(input_col);
    // Result: [out_channels, in_channels*kernel_h*kernel_w]

    return d_kernel;
}

} // namespace nn

#endif // IM2COL_H
