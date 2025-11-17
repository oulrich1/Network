#include "conv_layer.h"
#include <iostream>
#include <cassert>
#include <cmath>

// Use ml namespace types
using ml::ActivationType;

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, eps) assert(std::abs((a) - (b)) < (eps))

void test_conv2d_construction() {
    std::cout << "Testing Conv2D construction..." << std::endl;

    // Valid construction
    Conv2D<float> conv1(32, 3, 3, ActivationType::RELU);
    Conv2D<float> conv2(64, 5, 5, ActivationType::SIGMOID, 2, 2, 2, 2);

    // Test invalid parameters
    bool caught = false;
    try {
        Conv2D<float> conv_bad(0, 3, 3, ActivationType::RELU);  // Invalid out_channels
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Should catch invalid out_channels");

    caught = false;
    try {
        Conv2D<float> conv_bad(32, 3, 3, ActivationType::RELU, 0, 1);  // Invalid stride
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Should catch invalid stride");

    std::cout << "  ✓ Construction tests passed" << std::endl;
}

void test_conv2d_initialization() {
    std::cout << "Testing Conv2D initialization..." << std::endl;

    Conv2D<float> conv(16, 3, 3, ActivationType::RELU);
    conv.setInputChannels(3);
    conv.init();

    // Check kernel shape
    auto kernels = conv.getKernels();
    ASSERT_EQ(kernels.ndim(), 4);
    ASSERT_EQ(kernels.shape(0), 16);  // out_channels
    ASSERT_EQ(kernels.shape(1), 3);   // in_channels
    ASSERT_EQ(kernels.shape(2), 3);   // kernel_h
    ASSERT_EQ(kernels.shape(3), 3);   // kernel_w

    // Check bias shape
    auto bias = conv.getBias();
    ASSERT_EQ(bias.ndim(), 1);
    ASSERT_EQ(bias.shape(0), 16);

    // Kernels should be initialized with reasonable values (not all zeros)
    float kernel_sum = kernels.sum();
    assert(std::abs(kernel_sum) > 1e-6 && "Kernels should be initialized");

    // Test He initialization magnitude (should be ~ sqrt(2/fan_in))
    // fan_in = 3 * 3 * 3 = 27, so std ~ sqrt(2/27) ~ 0.27
    float kernel_mean = kernels.mean();
    float kernel_std = 0;
    for (size_t i = 0; i < kernels.size(); ++i) {
        kernel_std += (kernels(i) - kernel_mean) * (kernels(i) - kernel_mean);
    }
    kernel_std = std::sqrt(kernel_std / kernels.size());

    // Std should be roughly in the right ballpark (0.1 to 0.5 for this case)
    assert(kernel_std > 0.05 && kernel_std < 1.0 && "Kernel std in reasonable range");

    std::cout << "  ✓ Initialization tests passed" << std::endl;
}

void test_conv2d_forward_basic() {
    std::cout << "Testing Conv2D forward pass (basic)..." << std::endl;

    // Create simple 3x3 input with 1 channel
    Tensor<float> input({1, 1, 3, 3});
    input(0, 0, 0, 0) = 1; input(0, 0, 0, 1) = 2; input(0, 0, 0, 2) = 3;
    input(0, 0, 1, 0) = 4; input(0, 0, 1, 1) = 5; input(0, 0, 1, 2) = 6;
    input(0, 0, 2, 0) = 7; input(0, 0, 2, 1) = 8; input(0, 0, 2, 2) = 9;

    // Create conv layer with 1 filter, 2x2 kernel, no padding, stride 1
    Conv2D<float> conv(1, 2, 2, ActivationType::LINEAR);  // Linear activation for easier testing
    conv.setInputChannels(1);
    conv.init();

    // Set known weights for predictable output
    Tensor<float> kernel({1, 1, 2, 2});
    kernel(0, 0, 0, 0) = 1; kernel(0, 0, 0, 1) = 0;
    kernel(0, 0, 1, 0) = 0; kernel(0, 0, 1, 1) = 1;
    conv.setKernels(kernel);

    Tensor<float> bias({1});
    bias(0) = 0;
    conv.setBias(bias);

    // Forward pass
    auto output = conv.forward(input);

    // Check output shape
    ASSERT_EQ(output.ndim(), 4);
    ASSERT_EQ(output.shape(0), 1);  // batch
    ASSERT_EQ(output.shape(1), 1);  // out_channels
    ASSERT_EQ(output.shape(2), 2);  // out_h = (3 - 2) / 1 + 1 = 2
    ASSERT_EQ(output.shape(3), 2);  // out_w = 2

    // Check output values
    // Top-left: 1*1 + 2*0 + 4*0 + 5*1 = 1 + 5 = 6
    ASSERT_NEAR(output(0, 0, 0, 0), 6.0f, 1e-5f);

    // Top-right: 2*1 + 3*0 + 5*0 + 6*1 = 2 + 6 = 8
    ASSERT_NEAR(output(0, 0, 0, 1), 8.0f, 1e-5f);

    // Bottom-left: 4*1 + 5*0 + 7*0 + 8*1 = 4 + 8 = 12
    ASSERT_NEAR(output(0, 0, 1, 0), 12.0f, 1e-5f);

    // Bottom-right: 5*1 + 6*0 + 8*0 + 9*1 = 5 + 9 = 14
    ASSERT_NEAR(output(0, 0, 1, 1), 14.0f, 1e-5f);

    std::cout << "  ✓ Forward pass basic tests passed" << std::endl;
}

void test_conv2d_forward_with_padding() {
    std::cout << "Testing Conv2D forward pass with padding..." << std::endl;

    // 2x2 input
    Tensor<float> input({1, 1, 2, 2});
    input(0, 0, 0, 0) = 1; input(0, 0, 0, 1) = 2;
    input(0, 0, 1, 0) = 3; input(0, 0, 1, 1) = 4;

    // 3x3 kernel with padding=1
    Conv2D<float> conv(1, 3, 3, ActivationType::LINEAR, 1, 1, 1, 1);
    conv.setInputChannels(1);
    conv.init();

    auto output = conv.forward(input);

    // With padding=1, output should be same size as input (2x2)
    ASSERT_EQ(output.shape(2), 2);
    ASSERT_EQ(output.shape(3), 2);

    std::cout << "  ✓ Forward pass with padding tests passed" << std::endl;
}

void test_conv2d_forward_with_stride() {
    std::cout << "Testing Conv2D forward pass with stride..." << std::endl;

    // 4x4 input
    Tensor<float> input = Tensor<float>::ones({1, 1, 4, 4});

    // 2x2 kernel with stride=2
    Conv2D<float> conv(1, 2, 2, ActivationType::LINEAR, 2, 2, 0, 0);
    conv.setInputChannels(1);
    conv.init();

    auto output = conv.forward(input);

    // Output: (4 - 2) / 2 + 1 = 2
    ASSERT_EQ(output.shape(2), 2);
    ASSERT_EQ(output.shape(3), 2);

    std::cout << "  ✓ Forward pass with stride tests passed" << std::endl;
}

void test_conv2d_multi_channel() {
    std::cout << "Testing Conv2D with multiple channels..." << std::endl;

    // 3x3 input with 3 channels (RGB-like)
    Tensor<float> input = Tensor<float>::random({1, 3, 3, 3}, 0.0f, 1.0f);

    // 2 filters, 2x2 kernel
    Conv2D<float> conv(2, 2, 2, ActivationType::RELU);
    conv.setInputChannels(3);
    conv.init();

    auto output = conv.forward(input);

    // Check shape
    ASSERT_EQ(output.shape(0), 1);  // batch
    ASSERT_EQ(output.shape(1), 2);  // out_channels = 2
    ASSERT_EQ(output.shape(2), 2);  // out_h = (3 - 2) + 1 = 2
    ASSERT_EQ(output.shape(3), 2);  // out_w = 2

    std::cout << "  ✓ Multi-channel tests passed" << std::endl;
}

void test_conv2d_batch_processing() {
    std::cout << "Testing Conv2D with batch > 1..." << std::endl;

    // Batch of 4 images
    Tensor<float> input = Tensor<float>::random({4, 1, 5, 5}, 0.0f, 1.0f);

    Conv2D<float> conv(8, 3, 3, ActivationType::RELU);
    conv.setInputChannels(1);
    conv.init();

    auto output = conv.forward(input);

    // Check shape
    ASSERT_EQ(output.shape(0), 4);  // batch preserved
    ASSERT_EQ(output.shape(1), 8);  // out_channels
    ASSERT_EQ(output.shape(2), 3);  // (5 - 3) + 1 = 3
    ASSERT_EQ(output.shape(3), 3);

    std::cout << "  ✓ Batch processing tests passed" << std::endl;
}

void test_conv2d_activation_functions() {
    std::cout << "Testing Conv2D with different activations..." << std::endl;

    Tensor<float> input = Tensor<float>::random({1, 1, 4, 4}, -1.0f, 1.0f);

    // Test different activations
    ActivationType activations[] = {ActivationType::RELU, ActivationType::SIGMOID, ActivationType::TANH, ActivationType::LINEAR, ActivationType::LEAKY_RELU};
    const char* names[] = {"RELU", "ActivationType::SIGMOID", "ActivationType::TANH", "ActivationType::LINEAR", "ActivationType::LEAKY_RELU"};

    for (int i = 0; i < 5; ++i) {
        Conv2D<float> conv(4, 3, 3, activations[i]);
        conv.setInputChannels(1);
        conv.init();

        auto output = conv.forward(input);
        ASSERT_EQ(output.shape(1), 4);  // Should produce output

        // ReLU output should have no negative values
        if (activations[i] == ActivationType::RELU) {
            for (size_t j = 0; j < output.size(); ++j) {
                assert(output(j) >= 0.0f && "ReLU output should be non-negative");
            }
        }

        // Sigmoid output should be in [0, 1]
        if (activations[i] == ActivationType::SIGMOID) {
            for (size_t j = 0; j < output.size(); ++j) {
                assert(output(j) >= 0.0f && output(j) <= 1.0f && "Sigmoid in [0,1]");
            }
        }
    }

    std::cout << "  ✓ Activation function tests passed" << std::endl;
}

void test_conv2d_backward_numerical_gradient() {
    std::cout << "Testing Conv2D backward pass (numerical gradient)..." << std::endl;

    // Small network for gradient checking
    Tensor<float> input = Tensor<float>::random({1, 1, 4, 4}, 0.0f, 1.0f);

    Conv2D<float> conv(2, 3, 3, ActivationType::LINEAR);  // Linear for simpler gradients
    conv.setInputChannels(1);
    conv.init();

    // Forward pass
    auto output = conv.forward(input);

    // Create dummy gradient (all ones for simplicity)
    Tensor<float> d_output = Tensor<float>::ones(output.shape());

    // Backward pass
    auto d_input = conv.backward(d_output);

    // Check gradient shapes
    ASSERT_EQ(d_input.shape(), input.shape());

    auto kernel_grad = conv.getKernelGradients();
    ASSERT_EQ(kernel_grad.shape(0), 2);
    ASSERT_EQ(kernel_grad.shape(1), 1);
    ASSERT_EQ(kernel_grad.shape(2), 3);
    ASSERT_EQ(kernel_grad.shape(3), 3);

    auto bias_grad = conv.getBiasGradients();
    ASSERT_EQ(bias_grad.shape(0), 2);

    // Numerical gradient checking (simplified - just check one weight)
    float eps = 1e-4f;
    auto kernels = conv.getKernels();
    float original = kernels(0, 0, 0, 0);

    // f(x + eps)
    Tensor<float> kernels_plus = kernels.copy();
    kernels_plus(0, 0, 0, 0) = original + eps;
    conv.setKernels(kernels_plus);
    auto output_plus = conv.forward(input);
    float loss_plus = output_plus.sum();

    // f(x - eps)
    Tensor<float> kernels_minus = kernels.copy();
    kernels_minus(0, 0, 0, 0) = original - eps;
    conv.setKernels(kernels_minus);
    auto output_minus = conv.forward(input);
    float loss_minus = output_minus.sum();

    // Numerical gradient
    float numerical_grad = (loss_plus - loss_minus) / (2 * eps);

    // Restore and get analytical gradient
    conv.setKernels(kernels);
    conv.forward(input);
    conv.backward(d_output);
    float analytical_grad = conv.getKernelGradients()(0, 0, 0, 0);

    // They should be close
    float relative_error = std::abs(numerical_grad - analytical_grad) /
                          (std::abs(numerical_grad) + std::abs(analytical_grad) + 1e-8f);

    std::cout << "    Numerical gradient: " << numerical_grad << std::endl;
    std::cout << "    Analytical gradient: " << analytical_grad << std::endl;
    std::cout << "    Relative error: " << relative_error << std::endl;

    assert(relative_error < 1e-2f && "Gradient check failed");

    std::cout << "  ✓ Backward pass gradient tests passed" << std::endl;
}

void test_conv2d_weight_update() {
    std::cout << "Testing Conv2D weight updates..." << std::endl;

    Tensor<float> input = Tensor<float>::random({2, 1, 4, 4}, 0.0f, 1.0f);
    Tensor<float> target = Tensor<float>::random({2, 4, 2, 2}, 0.0f, 1.0f);

    Conv2D<float> conv(4, 3, 3, ActivationType::LINEAR);
    conv.setInputChannels(1);
    conv.init();

    auto kernels_before = conv.getKernels().copy();
    auto bias_before = conv.getBias().copy();

    // Forward-backward
    auto output = conv.forward(input);
    auto d_output = output - target;  // Simple loss gradient
    conv.backward(d_output);

    // Update weights
    float lr = 0.01f;
    conv.updateWeights(lr);

    auto kernels_after = conv.getKernels();
    auto bias_after = conv.getBias();

    // Weights should have changed
    bool kernels_changed = false;
    for (size_t i = 0; i < kernels_before.size(); ++i) {
        if (std::abs(kernels_before(i) - kernels_after(i)) > 1e-6f) {
            kernels_changed = true;
            break;
        }
    }
    assert(kernels_changed && "Kernels should be updated");

    bool bias_changed = false;
    for (size_t i = 0; i < bias_before.size(); ++i) {
        if (std::abs(bias_before(i) - bias_after(i)) > 1e-6f) {
            bias_changed = true;
            break;
        }
    }
    assert(bias_changed && "Bias should be updated");

    std::cout << "  ✓ Weight update tests passed" << std::endl;
}

void test_conv2d_mnist_like() {
    std::cout << "Testing Conv2D with MNIST-like dimensions..." << std::endl;

    // MNIST: 28x28 grayscale images
    Tensor<float> input = Tensor<float>::random({8, 1, 28, 28}, 0.0f, 1.0f);

    // First conv layer: 32 filters, 5x5 kernel
    Conv2D<float> conv1(32, 5, 5, ActivationType::RELU);
    conv1.setInputChannels(1);
    conv1.init();

    auto output1 = conv1.forward(input);

    // Output: (28 - 5) + 1 = 24
    ASSERT_EQ(output1.shape(0), 8);   // batch
    ASSERT_EQ(output1.shape(1), 32);  // channels
    ASSERT_EQ(output1.shape(2), 24);
    ASSERT_EQ(output1.shape(3), 24);

    std::cout << "  ✓ MNIST-like dimension tests passed" << std::endl;
}

int main() {
    std::cout << "\n=== Conv2D Test Suite ===" << std::endl;

    test_conv2d_construction();
    test_conv2d_initialization();
    test_conv2d_forward_basic();
    test_conv2d_forward_with_padding();
    test_conv2d_forward_with_stride();
    test_conv2d_multi_channel();
    test_conv2d_batch_processing();
    test_conv2d_activation_functions();
    test_conv2d_backward_numerical_gradient();
    test_conv2d_weight_update();
    test_conv2d_mnist_like();

    std::cout << "\n✓ All Conv2D tests passed!" << std::endl;
    std::cout << "\nConv2D layer is working correctly and ready for CNN networks." << std::endl;
    return 0;
}
