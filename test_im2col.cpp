#include "im2col.h"
#include <iostream>
#include <cassert>
#include <cmath>

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, eps) assert(std::abs((a) - (b)) < (eps))

void test_im2col_basic() {
    std::cout << "Testing im2col basic functionality..." << std::endl;

    // Create a simple 4x4 image with 1 channel, batch=1
    Tensor<float> input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) {
        input(i) = (float)(i + 1);  // 1, 2, 3, ..., 16
    }

    // Apply im2col with 3x3 kernel, stride=1, no padding
    // Expected output: 2x2 spatial positions, each with 9 values (3x3 patch)
    auto col = nn::im2col<float>(input, 3, 3, 1, 1, 0, 0);

    auto col_size = col.size();
    ASSERT_EQ(col_size.cy, 4);  // 1 * 2 * 2 = 4 patches
    ASSERT_EQ(col_size.cx, 9);  // 1 * 3 * 3 = 9 values per patch

    // Check first patch (top-left 3x3)
    // Should be: 1, 2, 3, 5, 6, 7, 9, 10, 11
    ASSERT_NEAR(col.getAt(0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 1), 2.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 2), 3.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 3), 5.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 4), 6.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 5), 7.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 6), 9.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 7), 10.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 8), 11.0f, 1e-6f);

    // Check second patch (top-right 3x3, shifted by 1)
    // Should be: 2, 3, 4, 6, 7, 8, 10, 11, 12
    ASSERT_NEAR(col.getAt(1, 0), 2.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(1, 1), 3.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(1, 2), 4.0f, 1e-6f);

    std::cout << "  ✓ im2col basic test passed" << std::endl;
}

void test_im2col_with_padding() {
    std::cout << "Testing im2col with padding..." << std::endl;

    // 2x2 image
    Tensor<float> input({1, 1, 2, 2});
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 1, 0) = 3.0f;
    input(0, 0, 1, 1) = 4.0f;

    // 3x3 kernel with padding=1
    // With padding, input becomes:
    // [0 0 0 0]
    // [0 1 2 0]
    // [0 3 4 0]
    // [0 0 0 0]
    // Output: 2x2 patches
    auto col = nn::im2col<float>(input, 3, 3, 1, 1, 1, 1);

    auto col_size = col.size();
    ASSERT_EQ(col_size.cy, 4);  // 1 * 2 * 2 = 4 patches
    ASSERT_EQ(col_size.cx, 9);  // 3 * 3 = 9

    // First patch (top-left, mostly padding)
    // Should be: 0, 0, 0, 0, 1, 2, 0, 3, 4
    ASSERT_NEAR(col.getAt(0, 0), 0.0f, 1e-6f);  // padding
    ASSERT_NEAR(col.getAt(0, 4), 1.0f, 1e-6f);  // center of patch
    ASSERT_NEAR(col.getAt(0, 5), 2.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 7), 3.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 8), 4.0f, 1e-6f);

    std::cout << "  ✓ im2col with padding test passed" << std::endl;
}

void test_im2col_with_stride() {
    std::cout << "Testing im2col with stride > 1..." << std::endl;

    // 4x4 image
    Tensor<float> input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) {
        input(i) = (float)(i + 1);
    }

    // 2x2 kernel with stride=2
    // Should extract 4 non-overlapping 2x2 patches
    auto col = nn::im2col<float>(input, 2, 2, 2, 2, 0, 0);

    auto col_size = col.size();
    ASSERT_EQ(col_size.cy, 4);  // 1 * 2 * 2 = 4 patches (2x2 grid with stride 2)
    ASSERT_EQ(col_size.cx, 4);  // 2 * 2 = 4 values per patch

    // First patch: top-left 2x2 = [1, 2, 5, 6]
    ASSERT_NEAR(col.getAt(0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 1), 2.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 2), 5.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 3), 6.0f, 1e-6f);

    // Second patch: top-right 2x2 = [3, 4, 7, 8]
    ASSERT_NEAR(col.getAt(1, 0), 3.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(1, 1), 4.0f, 1e-6f);

    // Fourth patch: bottom-right 2x2 = [11, 12, 15, 16]
    ASSERT_NEAR(col.getAt(3, 0), 11.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(3, 1), 12.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(3, 2), 15.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(3, 3), 16.0f, 1e-6f);

    std::cout << "  ✓ im2col with stride test passed" << std::endl;
}

void test_im2col_multi_channel() {
    std::cout << "Testing im2col with multiple channels..." << std::endl;

    // 3x3 image with 2 channels
    Tensor<float> input({1, 2, 3, 3});

    // Channel 0
    for (int i = 0; i < 9; ++i) {
        input(0, 0, i / 3, i % 3) = (float)(i + 1);
    }
    // Channel 1
    for (int i = 0; i < 9; ++i) {
        input(0, 1, i / 3, i % 3) = (float)((i + 1) * 10);
    }

    // 2x2 kernel, no stride, no padding
    auto col = nn::im2col<float>(input, 2, 2, 1, 1, 0, 0);

    auto col_size = col.size();
    ASSERT_EQ(col_size.cy, 4);   // 1 * 2 * 2 = 4 patches
    ASSERT_EQ(col_size.cx, 8);   // 2 channels * 2 * 2 = 8

    // First patch, channel 0: [1, 2, 4, 5]
    ASSERT_NEAR(col.getAt(0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 1), 2.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 2), 4.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 3), 5.0f, 1e-6f);

    // First patch, channel 1: [10, 20, 40, 50]
    ASSERT_NEAR(col.getAt(0, 4), 10.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 5), 20.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 6), 40.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 7), 50.0f, 1e-6f);

    std::cout << "  ✓ im2col multi-channel test passed" << std::endl;
}

void test_col2im_basic() {
    std::cout << "Testing col2im basic functionality..." << std::endl;

    // Create column matrix (2x2 patches with 2x2 values each)
    ml::Mat<float> col(4, 4);
    // Patch 0: all 1s
    for (int j = 0; j < 4; ++j) col.setAt(0, j, 1.0f);
    // Patch 1: all 2s
    for (int j = 0; j < 4; ++j) col.setAt(1, j, 2.0f);
    // Patch 2: all 3s
    for (int j = 0; j < 4; ++j) col.setAt(2, j, 3.0f);
    // Patch 3: all 4s
    for (int j = 0; j < 4; ++j) col.setAt(3, j, 4.0f);

    // Convert back to image (4x4 with stride=2, no overlap)
    auto output = nn::col2im<float>(col, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);

    ASSERT_EQ(output.ndim(), 4);
    ASSERT_EQ(output.shape(0), 1);  // batch
    ASSERT_EQ(output.shape(1), 1);  // channels
    ASSERT_EQ(output.shape(2), 4);  // height
    ASSERT_EQ(output.shape(3), 4);  // width

    // With non-overlapping patches, each position gets value from one patch
    ASSERT_NEAR(output(0, 0, 0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(output(0, 0, 0, 1), 1.0f, 1e-6f);
    ASSERT_NEAR(output(0, 0, 0, 2), 2.0f, 1e-6f);
    ASSERT_NEAR(output(0, 0, 0, 3), 2.0f, 1e-6f);

    ASSERT_NEAR(output(0, 0, 2, 0), 3.0f, 1e-6f);
    ASSERT_NEAR(output(0, 0, 3, 3), 4.0f, 1e-6f);

    std::cout << "  ✓ col2im basic test passed" << std::endl;
}

void test_im2col_col2im_roundtrip() {
    std::cout << "Testing im2col → col2im round-trip..." << std::endl;

    // Create original image
    Tensor<float> input({1, 1, 4, 4});
    for (int i = 0; i < 16; ++i) {
        input(i) = (float)(i + 1);
    }

    // im2col with non-overlapping patches (stride = kernel size)
    auto col = nn::im2col<float>(input, 2, 2, 2, 2, 0, 0);

    // col2im back to image
    auto output = nn::col2im<float>(col, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);

    // Should match original
    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_NEAR(input(i), output(i), 1e-6f);
    }

    std::cout << "  ✓ im2col/col2im round-trip test passed" << std::endl;
}

void test_im2col_col2im_with_overlap() {
    std::cout << "Testing im2col → col2im with overlapping patches..." << std::endl;

    // Create image
    Tensor<float> input({1, 1, 3, 3}, 1.0f);  // All ones

    // im2col with overlapping patches (2x2 kernel, stride=1)
    auto col = nn::im2col<float>(input, 2, 2, 1, 1, 0, 0);

    // col2im back - values will accumulate where patches overlap
    auto output = nn::col2im<float>(col, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0);

    // Corner pixels appear in 1 patch each
    ASSERT_NEAR(output(0, 0, 0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(output(0, 0, 0, 2), 1.0f, 1e-6f);
    ASSERT_NEAR(output(0, 0, 2, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(output(0, 0, 2, 2), 1.0f, 1e-6f);

    // Edge pixels appear in 2 patches each
    ASSERT_NEAR(output(0, 0, 0, 1), 2.0f, 1e-6f);
    ASSERT_NEAR(output(0, 0, 1, 0), 2.0f, 1e-6f);

    // Center pixel appears in 4 patches
    ASSERT_NEAR(output(0, 0, 1, 1), 4.0f, 1e-6f);

    std::cout << "  ✓ im2col/col2im with overlap test passed" << std::endl;
}

void test_output_dims_calculation() {
    std::cout << "Testing output dimensions calculation..." << std::endl;

    int out_h, out_w;

    // Basic case: 28x28 input, 5x5 kernel, stride 1, no padding
    nn::im2col_get_output_dims<float>(28, 28, 5, 5, 1, 1, 0, 0, out_h, out_w);
    ASSERT_EQ(out_h, 24);
    ASSERT_EQ(out_w, 24);

    // With padding: 28x28 input, 5x5 kernel, stride 1, padding 2
    nn::im2col_get_output_dims<float>(28, 28, 5, 5, 1, 1, 2, 2, out_h, out_w);
    ASSERT_EQ(out_h, 28);  // Same as input (padding preserves size)
    ASSERT_EQ(out_w, 28);

    // With stride: 28x28 input, 3x3 kernel, stride 2, padding 1
    nn::im2col_get_output_dims<float>(28, 28, 3, 3, 2, 2, 1, 1, out_h, out_w);
    ASSERT_EQ(out_h, 14);
    ASSERT_EQ(out_w, 14);

    std::cout << "  ✓ Output dimensions calculation test passed" << std::endl;
}

void test_batch_processing() {
    std::cout << "Testing im2col with batch > 1..." << std::endl;

    // Create batch of 2 images
    Tensor<float> input({2, 1, 3, 3});

    // Batch 0: values 1-9
    for (int i = 0; i < 9; ++i) {
        input(0, 0, i / 3, i % 3) = (float)(i + 1);
    }

    // Batch 1: values 10-18
    for (int i = 0; i < 9; ++i) {
        input(1, 0, i / 3, i % 3) = (float)(i + 10);
    }

    // 2x2 kernel, stride 1
    auto col = nn::im2col<float>(input, 2, 2, 1, 1, 0, 0);

    auto col_size = col.size();
    ASSERT_EQ(col_size.cy, 8);  // 2 batches * 2 * 2 = 8 patches
    ASSERT_EQ(col_size.cx, 4);  // 2 * 2 = 4 values per patch

    // First patch of batch 0: [1, 2, 4, 5]
    ASSERT_NEAR(col.getAt(0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(0, 3), 5.0f, 1e-6f);

    // First patch of batch 1 (5th overall patch): [10, 11, 13, 14]
    ASSERT_NEAR(col.getAt(4, 0), 10.0f, 1e-6f);
    ASSERT_NEAR(col.getAt(4, 3), 14.0f, 1e-6f);

    std::cout << "  ✓ Batch processing test passed" << std::endl;
}

int main() {
    std::cout << "\n=== im2col/col2im Test Suite ===" << std::endl;

    test_im2col_basic();
    test_im2col_with_padding();
    test_im2col_with_stride();
    test_im2col_multi_channel();
    test_col2im_basic();
    test_im2col_col2im_roundtrip();
    test_im2col_col2im_with_overlap();
    test_output_dims_calculation();
    test_batch_processing();

    std::cout << "\n✓ All im2col/col2im tests passed!" << std::endl;
    std::cout << "\nim2col/col2im is working correctly and ready for CNN implementation." << std::endl;
    return 0;
}
