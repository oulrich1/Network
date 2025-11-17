#include "pooling_layer.h"
#include <iostream>
#include <cassert>
#include <cmath>

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, eps) assert(std::abs((a) - (b)) < (eps))

void test_maxpool_construction() {
    std::cout << "Testing MaxPool2D construction..." << std::endl;

    MaxPool2D<float> pool1(2, 2, 2, 2);  // Explicit stride
    MaxPool2D<float> pool2(3, 3, 2, 2);
    MaxPool2D<float> pool3(2);  // Square pooling convenience constructor

    // Test invalid parameters
    bool caught = false;
    try {
        MaxPool2D<float> pool_bad(0);  // Invalid pool size
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Should catch invalid pool size");

    std::cout << "  ✓ Construction tests passed" << std::endl;
}

void test_maxpool_forward_basic() {
    std::cout << "Testing MaxPool2D forward pass (basic)..." << std::endl;

    // Create 4x4 input with known values
    Tensor<float> input({1, 1, 4, 4});
    input(0, 0, 0, 0) = 1;  input(0, 0, 0, 1) = 2;  input(0, 0, 0, 2) = 3;  input(0, 0, 0, 3) = 4;
    input(0, 0, 1, 0) = 5;  input(0, 0, 1, 1) = 6;  input(0, 0, 1, 2) = 7;  input(0, 0, 1, 3) = 8;
    input(0, 0, 2, 0) = 9;  input(0, 0, 2, 1) = 10; input(0, 0, 2, 2) = 11; input(0, 0, 2, 3) = 12;
    input(0, 0, 3, 0) = 13; input(0, 0, 3, 1) = 14; input(0, 0, 3, 2) = 15; input(0, 0, 3, 3) = 16;

    // 2x2 pooling with stride 2 (non-overlapping)
    MaxPool2D<float> pool(2, 2, 2, 2);
    auto output = pool.forward(input);

    // Check output shape
    ASSERT_EQ(output.ndim(), 4);
    ASSERT_EQ(output.shape(0), 1);  // batch
    ASSERT_EQ(output.shape(1), 1);  // channels
    ASSERT_EQ(output.shape(2), 2);  // (4 - 2) / 2 + 1 = 2
    ASSERT_EQ(output.shape(3), 2);

    // Check output values (max in each 2x2 window)
    // Top-left: max(1,2,5,6) = 6
    ASSERT_NEAR(output(0, 0, 0, 0), 6.0f, 1e-6f);
    // Top-right: max(3,4,7,8) = 8
    ASSERT_NEAR(output(0, 0, 0, 1), 8.0f, 1e-6f);
    // Bottom-left: max(9,10,13,14) = 14
    ASSERT_NEAR(output(0, 0, 1, 0), 14.0f, 1e-6f);
    // Bottom-right: max(11,12,15,16) = 16
    ASSERT_NEAR(output(0, 0, 1, 1), 16.0f, 1e-6f);

    std::cout << "  ✓ MaxPool forward basic tests passed" << std::endl;
}

void test_maxpool_backward() {
    std::cout << "Testing MaxPool2D backward pass..." << std::endl;

    // Simple 2x2 input
    Tensor<float> input({1, 1, 2, 2});
    input(0, 0, 0, 0) = 1;
    input(0, 0, 0, 1) = 4;  // Max
    input(0, 0, 1, 0) = 3;
    input(0, 0, 1, 1) = 2;

    MaxPool2D<float> pool(2);
    auto output = pool.forward(input);

    // Gradient w.r.t. output (all ones)
    Tensor<float> d_output = Tensor<float>::ones(output.shape());

    // Backward pass
    auto d_input = pool.backward(d_output);

    // Gradient should only flow to the max position (0, 1)
    ASSERT_NEAR(d_input(0, 0, 0, 0), 0.0f, 1e-6f);
    ASSERT_NEAR(d_input(0, 0, 0, 1), 1.0f, 1e-6f);  // Max position
    ASSERT_NEAR(d_input(0, 0, 1, 0), 0.0f, 1e-6f);
    ASSERT_NEAR(d_input(0, 0, 1, 1), 0.0f, 1e-6f);

    std::cout << "  ✓ MaxPool backward tests passed" << std::endl;
}

void test_maxpool_overlapping() {
    std::cout << "Testing MaxPool2D with overlapping windows..." << std::endl;

    // 3x3 input
    Tensor<float> input({1, 1, 3, 3});
    for (int i = 0; i < 9; ++i) {
        input(0, 0, i / 3, i % 3) = (float)(i + 1);
    }

    // 2x2 pool with stride 1 (overlapping)
    MaxPool2D<float> pool(2, 2, 1, 1);
    auto output = pool.forward(input);

    // Output should be 2x2
    ASSERT_EQ(output.shape(2), 2);
    ASSERT_EQ(output.shape(3), 2);

    // Top-left: max(1,2,4,5) = 5
    ASSERT_NEAR(output(0, 0, 0, 0), 5.0f, 1e-6f);
    // Top-right: max(2,3,5,6) = 6
    ASSERT_NEAR(output(0, 0, 0, 1), 6.0f, 1e-6f);
    // Bottom-left: max(4,5,7,8) = 8
    ASSERT_NEAR(output(0, 0, 1, 0), 8.0f, 1e-6f);
    // Bottom-right: max(5,6,8,9) = 9
    ASSERT_NEAR(output(0, 0, 1, 1), 9.0f, 1e-6f);

    std::cout << "  ✓ MaxPool overlapping tests passed" << std::endl;
}

void test_maxpool_multi_channel() {
    std::cout << "Testing MaxPool2D with multiple channels..." << std::endl;

    // 4x4 input with 3 channels
    Tensor<float> input = Tensor<float>::random({1, 3, 4, 4}, 0.0f, 10.0f);

    MaxPool2D<float> pool(2);
    auto output = pool.forward(input);

    // Channels should be preserved
    ASSERT_EQ(output.shape(1), 3);
    ASSERT_EQ(output.shape(2), 2);
    ASSERT_EQ(output.shape(3), 2);

    std::cout << "  ✓ MaxPool multi-channel tests passed" << std::endl;
}

void test_maxpool_batch() {
    std::cout << "Testing MaxPool2D with batch > 1..." << std::endl;

    // Batch of 4 images
    Tensor<float> input = Tensor<float>::random({4, 2, 8, 8}, 0.0f, 1.0f);

    MaxPool2D<float> pool(2);
    auto output = pool.forward(input);

    // Batch should be preserved
    ASSERT_EQ(output.shape(0), 4);
    ASSERT_EQ(output.shape(1), 2);
    ASSERT_EQ(output.shape(2), 4);
    ASSERT_EQ(output.shape(3), 4);

    std::cout << "  ✓ MaxPool batch tests passed" << std::endl;
}

void test_avgpool_forward_basic() {
    std::cout << "Testing AvgPool2D forward pass (basic)..." << std::endl;

    // Create 4x4 input with known values
    Tensor<float> input({1, 1, 4, 4});
    input(0, 0, 0, 0) = 1;  input(0, 0, 0, 1) = 2;  input(0, 0, 0, 2) = 3;  input(0, 0, 0, 3) = 4;
    input(0, 0, 1, 0) = 5;  input(0, 0, 1, 1) = 6;  input(0, 0, 1, 2) = 7;  input(0, 0, 1, 3) = 8;
    input(0, 0, 2, 0) = 9;  input(0, 0, 2, 1) = 10; input(0, 0, 2, 2) = 11; input(0, 0, 2, 3) = 12;
    input(0, 0, 3, 0) = 13; input(0, 0, 3, 1) = 14; input(0, 0, 3, 2) = 15; input(0, 0, 3, 3) = 16;

    // 2x2 pooling with stride 2
    AvgPool2D<float> pool(2, 2, 2, 2);
    auto output = pool.forward(input);

    // Check output shape
    ASSERT_EQ(output.shape(2), 2);
    ASSERT_EQ(output.shape(3), 2);

    // Check output values (average in each 2x2 window)
    // Top-left: avg(1,2,5,6) = 3.5
    ASSERT_NEAR(output(0, 0, 0, 0), 3.5f, 1e-6f);
    // Top-right: avg(3,4,7,8) = 5.5
    ASSERT_NEAR(output(0, 0, 0, 1), 5.5f, 1e-6f);
    // Bottom-left: avg(9,10,13,14) = 11.5
    ASSERT_NEAR(output(0, 0, 1, 0), 11.5f, 1e-6f);
    // Bottom-right: avg(11,12,15,16) = 13.5
    ASSERT_NEAR(output(0, 0, 1, 1), 13.5f, 1e-6f);

    std::cout << "  ✓ AvgPool forward basic tests passed" << std::endl;
}

void test_avgpool_backward() {
    std::cout << "Testing AvgPool2D backward pass..." << std::endl;

    // Simple 2x2 input
    Tensor<float> input({1, 1, 2, 2});
    input(0, 0, 0, 0) = 1;
    input(0, 0, 0, 1) = 2;
    input(0, 0, 1, 0) = 3;
    input(0, 0, 1, 1) = 4;

    AvgPool2D<float> pool(2);
    auto output = pool.forward(input);

    // Gradient w.r.t. output (value 4)
    Tensor<float> d_output({1, 1, 1, 1}, 4.0f);

    // Backward pass
    auto d_input = pool.backward(d_output);

    // Gradient should be distributed evenly (4 / 4 = 1.0 to each position)
    ASSERT_NEAR(d_input(0, 0, 0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(d_input(0, 0, 0, 1), 1.0f, 1e-6f);
    ASSERT_NEAR(d_input(0, 0, 1, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(d_input(0, 0, 1, 1), 1.0f, 1e-6f);

    std::cout << "  ✓ AvgPool backward tests passed" << std::endl;
}

void test_global_avgpool() {
    std::cout << "Testing GlobalAvgPool2D..." << std::endl;

    // 4x4 input with 2 channels
    Tensor<float> input({1, 2, 4, 4});

    // Channel 0: all 1s
    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = 1.0f;
        }
    }

    // Channel 1: all 2s
    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            input(0, 1, h, w) = 2.0f;
        }
    }

    GlobalAvgPool2D<float> gap;
    auto output = gap.forward(input);

    // Output should be [1, 2, 1, 1]
    ASSERT_EQ(output.shape(0), 1);
    ASSERT_EQ(output.shape(1), 2);
    ASSERT_EQ(output.shape(2), 1);
    ASSERT_EQ(output.shape(3), 1);

    // Channel 0 average: 1.0
    ASSERT_NEAR(output(0, 0, 0, 0), 1.0f, 1e-6f);
    // Channel 1 average: 2.0
    ASSERT_NEAR(output(0, 1, 0, 0), 2.0f, 1e-6f);

    std::cout << "  ✓ GlobalAvgPool tests passed" << std::endl;
}

void test_pooling_preserves_channels() {
    std::cout << "Testing that pooling preserves channel count..." << std::endl;

    Tensor<float> input = Tensor<float>::random({2, 64, 14, 14}, 0.0f, 1.0f);

    MaxPool2D<float> maxpool(2);
    auto max_output = maxpool.forward(input);

    ASSERT_EQ(max_output.shape(0), 2);   // batch
    ASSERT_EQ(max_output.shape(1), 64);  // channels preserved
    ASSERT_EQ(max_output.shape(2), 7);   // spatial reduced
    ASSERT_EQ(max_output.shape(3), 7);

    AvgPool2D<float> avgpool(2);
    auto avg_output = avgpool.forward(input);

    ASSERT_EQ(avg_output.shape(0), 2);
    ASSERT_EQ(avg_output.shape(1), 64);
    ASSERT_EQ(avg_output.shape(2), 7);
    ASSERT_EQ(avg_output.shape(3), 7);

    std::cout << "  ✓ Channel preservation tests passed" << std::endl;
}

void test_pooling_mnist_like() {
    std::cout << "Testing pooling with MNIST-like dimensions..." << std::endl;

    // After a 5x5 conv on 28x28 MNIST, we get 24x24
    Tensor<float> input = Tensor<float>::random({8, 32, 24, 24}, 0.0f, 1.0f);

    MaxPool2D<float> pool(2);
    auto output = pool.forward(input);

    // After 2x2 pooling: 24/2 = 12
    ASSERT_EQ(output.shape(0), 8);
    ASSERT_EQ(output.shape(1), 32);
    ASSERT_EQ(output.shape(2), 12);
    ASSERT_EQ(output.shape(3), 12);

    // Test backward pass
    Tensor<float> d_output = Tensor<float>::ones(output.shape());
    auto d_input = pool.backward(d_output);

    ASSERT_EQ(d_input.shape(), input.shape());

    std::cout << "  ✓ MNIST-like pooling tests passed" << std::endl;
}

int main() {
    std::cout << "\n=== Pooling Layer Test Suite ===" << std::endl;

    // MaxPool tests
    test_maxpool_construction();
    test_maxpool_forward_basic();
    test_maxpool_backward();
    test_maxpool_overlapping();
    test_maxpool_multi_channel();
    test_maxpool_batch();

    // AvgPool tests
    test_avgpool_forward_basic();
    test_avgpool_backward();

    // GlobalAvgPool tests
    test_global_avgpool();

    // General tests
    test_pooling_preserves_channels();
    test_pooling_mnist_like();

    std::cout << "\n✓ All pooling layer tests passed!" << std::endl;
    std::cout << "\nPooling layers are working correctly and ready for CNN networks." << std::endl;
    return 0;
}
