#include "tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, eps) assert(std::abs((a) - (b)) < (eps))

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;

    // Test basic construction
    Tensor<float> t1({2, 3, 4});
    ASSERT_EQ(t1.ndim(), 3);
    ASSERT_EQ(t1.shape(0), 2);
    ASSERT_EQ(t1.shape(1), 3);
    ASSERT_EQ(t1.shape(2), 4);
    ASSERT_EQ(t1.size(), 24);

    // Test zeros
    auto t2 = Tensor<float>::zeros({3, 4});
    ASSERT_EQ(t2.size(), 12);
    ASSERT_NEAR(t2.sum(), 0.0f, 1e-6f);

    // Test ones
    auto t3 = Tensor<float>::ones({2, 5});
    ASSERT_EQ(t3.size(), 10);
    ASSERT_NEAR(t3.sum(), 10.0f, 1e-6f);

    // Test fill
    Tensor<float> t4({3, 3}, 5.0f);
    ASSERT_NEAR(t4.mean(), 5.0f, 1e-6f);

    std::cout << "  ✓ Creation tests passed" << std::endl;
}

void test_tensor_indexing() {
    std::cout << "Testing tensor indexing..." << std::endl;

    // 2D indexing
    Tensor<float> t2d({2, 3});
    t2d(0, 0) = 1.0f;
    t2d(0, 1) = 2.0f;
    t2d(0, 2) = 3.0f;
    t2d(1, 0) = 4.0f;
    t2d(1, 1) = 5.0f;
    t2d(1, 2) = 6.0f;

    ASSERT_NEAR(t2d(0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(t2d(1, 2), 6.0f, 1e-6f);
    ASSERT_NEAR(t2d.sum(), 21.0f, 1e-6f);

    // 3D indexing
    Tensor<float> t3d({2, 3, 4});
    t3d(0, 0, 0) = 1.0f;
    t3d(1, 2, 3) = 99.0f;
    ASSERT_NEAR(t3d(0, 0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(t3d(1, 2, 3), 99.0f, 1e-6f);

    // 4D indexing
    Tensor<float> t4d({2, 3, 4, 5});
    t4d(0, 0, 0, 0) = 10.0f;
    t4d(1, 2, 3, 4) = 20.0f;
    ASSERT_NEAR(t4d(0, 0, 0, 0), 10.0f, 1e-6f);
    ASSERT_NEAR(t4d(1, 2, 3, 4), 20.0f, 1e-6f);

    std::cout << "  ✓ Indexing tests passed" << std::endl;
}

void test_tensor_reshape() {
    std::cout << "Testing tensor reshape..." << std::endl;

    Tensor<float> t({2, 3, 4});
    for (size_t i = 0; i < t.size(); ++i) {
        t(i) = (float)i;
    }

    // Reshape to 2D
    auto t2d = t.reshape({6, 4});
    ASSERT_EQ(t2d.ndim(), 2);
    ASSERT_EQ(t2d.shape(0), 6);
    ASSERT_EQ(t2d.shape(1), 4);
    ASSERT_NEAR(t2d(0, 0), 0.0f, 1e-6f);
    ASSERT_NEAR(t2d(5, 3), 23.0f, 1e-6f);

    // Reshape to 1D
    auto t1d = t.flatten();
    ASSERT_EQ(t1d.ndim(), 1);
    ASSERT_EQ(t1d.size(), 24);
    ASSERT_NEAR(t1d(0), 0.0f, 1e-6f);
    ASSERT_NEAR(t1d(23), 23.0f, 1e-6f);

    // Reshape back to 3D
    auto t3d = t1d.reshape({2, 3, 4});
    ASSERT_EQ(t3d.ndim(), 3);
    ASSERT_NEAR(t3d(0, 0, 0), 0.0f, 1e-6f);
    ASSERT_NEAR(t3d(1, 2, 3), 23.0f, 1e-6f);

    std::cout << "  ✓ Reshape tests passed" << std::endl;
}

void test_tensor_operations() {
    std::cout << "Testing tensor operations..." << std::endl;

    Tensor<float> t1({2, 3}, 2.0f);
    Tensor<float> t2({2, 3}, 3.0f);

    // Addition
    auto t3 = t1 + t2;
    ASSERT_NEAR(t3.mean(), 5.0f, 1e-6f);

    // Subtraction
    auto t4 = t2 - t1;
    ASSERT_NEAR(t4.mean(), 1.0f, 1e-6f);

    // Element-wise multiplication
    auto t5 = t1 * t2;
    ASSERT_NEAR(t5.mean(), 6.0f, 1e-6f);

    // Scalar multiplication
    auto t6 = t1 * 5.0f;
    ASSERT_NEAR(t6.mean(), 10.0f, 1e-6f);

    // Scalar division
    auto t7 = t2 / 3.0f;
    ASSERT_NEAR(t7.mean(), 1.0f, 1e-6f);

    // In-place operations
    Tensor<float> t8({2, 2}, 1.0f);
    t8 += Tensor<float>({2, 2}, 2.0f);
    ASSERT_NEAR(t8.mean(), 3.0f, 1e-6f);

    t8 *= 2.0f;
    ASSERT_NEAR(t8.mean(), 6.0f, 1e-6f);

    std::cout << "  ✓ Operations tests passed" << std::endl;
}

void test_tensor_transpose() {
    std::cout << "Testing tensor transpose..." << std::endl;

    // 2D transpose
    Tensor<float> t({2, 3});
    t(0, 0) = 1; t(0, 1) = 2; t(0, 2) = 3;
    t(1, 0) = 4; t(1, 1) = 5; t(1, 2) = 6;

    auto tT = t.transpose();
    ASSERT_EQ(tT.shape(0), 3);
    ASSERT_EQ(tT.shape(1), 2);
    ASSERT_NEAR(tT(0, 0), 1.0f, 1e-6f);
    ASSERT_NEAR(tT(0, 1), 4.0f, 1e-6f);
    ASSERT_NEAR(tT(2, 1), 6.0f, 1e-6f);

    // 4D transpose (NCHW → NHWC)
    Tensor<float> t4d({2, 3, 4, 5});
    for (size_t i = 0; i < t4d.size(); ++i) {
        t4d(i) = (float)i;
    }

    auto t_nhwc = t4d.transpose({0, 2, 3, 1});
    ASSERT_EQ(t_nhwc.shape(0), 2);  // N
    ASSERT_EQ(t_nhwc.shape(1), 4);  // H
    ASSERT_EQ(t_nhwc.shape(2), 5);  // W
    ASSERT_EQ(t_nhwc.shape(3), 3);  // C

    // Verify a specific element
    float original_val = t4d(0, 0, 0, 0);
    float transposed_val = t_nhwc(0, 0, 0, 0);
    ASSERT_NEAR(original_val, transposed_val, 1e-6f);

    std::cout << "  ✓ Transpose tests passed" << std::endl;
}

void test_tensor_mat_interop() {
    std::cout << "Testing Tensor <-> Mat interop..." << std::endl;

    // Create Mat
    ml::Mat<float> mat(3, 4);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            mat.setAt(i, j, (float)(i * 4 + j));
        }
    }

    // Mat → Tensor
    auto tensor = Tensor<float>::fromMat(mat, {3, 4});
    ASSERT_EQ(tensor.ndim(), 2);
    ASSERT_EQ(tensor.shape(0), 3);
    ASSERT_EQ(tensor.shape(1), 4);
    ASSERT_NEAR(tensor(0, 0), 0.0f, 1e-6f);
    ASSERT_NEAR(tensor(2, 3), 11.0f, 1e-6f);

    // Can also reshape to different dimensions
    auto tensor_reshaped = Tensor<float>::fromMat(mat, {2, 6});
    ASSERT_EQ(tensor_reshaped.shape(0), 2);
    ASSERT_EQ(tensor_reshaped.shape(1), 6);

    // Tensor → Mat
    auto mat2 = tensor.toMat();
    auto mat2Size = mat2.size();
    ASSERT_EQ(mat2Size.cy, 3);  // cy is height
    ASSERT_EQ(mat2Size.cx, 4);  // cx is width
    ASSERT_NEAR(mat2.getAt(0, 0), 0.0f, 1e-6f);
    ASSERT_NEAR(mat2.getAt(2, 3), 11.0f, 1e-6f);

    std::cout << "  ✓ Mat interop tests passed" << std::endl;
}

void test_tensor_squeeze_unsqueeze() {
    std::cout << "Testing squeeze/unsqueeze..." << std::endl;

    // Unsqueeze
    Tensor<float> t({3, 4});
    auto t_unsqueezed = t.unsqueeze(0);
    ASSERT_EQ(t_unsqueezed.ndim(), 3);
    ASSERT_EQ(t_unsqueezed.shape(0), 1);
    ASSERT_EQ(t_unsqueezed.shape(1), 3);
    ASSERT_EQ(t_unsqueezed.shape(2), 4);

    // Squeeze
    Tensor<float> t2({1, 3, 1, 4});
    auto t_squeezed = t2.squeeze();
    ASSERT_EQ(t_squeezed.ndim(), 2);
    ASSERT_EQ(t_squeezed.shape(0), 3);
    ASSERT_EQ(t_squeezed.shape(1), 4);

    std::cout << "  ✓ Squeeze/unsqueeze tests passed" << std::endl;
}

void test_tensor_statistics() {
    std::cout << "Testing tensor statistics..." << std::endl;

    Tensor<float> t({10}, 5.0f);
    ASSERT_NEAR(t.sum(), 50.0f, 1e-6f);
    ASSERT_NEAR(t.mean(), 5.0f, 1e-6f);
    ASSERT_NEAR(t.max(), 5.0f, 1e-6f);
    ASSERT_NEAR(t.min(), 5.0f, 1e-6f);

    // Mixed values
    Tensor<float> t2({5});
    t2(0) = 1.0f;
    t2(1) = 2.0f;
    t2(2) = 3.0f;
    t2(3) = 4.0f;
    t2(4) = 5.0f;

    ASSERT_NEAR(t2.sum(), 15.0f, 1e-6f);
    ASSERT_NEAR(t2.mean(), 3.0f, 1e-6f);
    ASSERT_NEAR(t2.max(), 5.0f, 1e-6f);
    ASSERT_NEAR(t2.min(), 1.0f, 1e-6f);

    std::cout << "  ✓ Statistics tests passed" << std::endl;
}

void test_tensor_copy() {
    std::cout << "Testing tensor copy..." << std::endl;

    Tensor<float> t1({2, 3}, 5.0f);
    auto t2 = t1.copy();

    // Verify copy
    ASSERT_EQ(t2.ndim(), t1.ndim());
    ASSERT_EQ(t2.size(), t1.size());
    ASSERT_NEAR(t2.mean(), 5.0f, 1e-6f);

    // Modify copy
    t2.fill(10.0f);
    ASSERT_NEAR(t2.mean(), 10.0f, 1e-6f);
    ASSERT_NEAR(t1.mean(), 5.0f, 1e-6f);  // Original unchanged

    std::cout << "  ✓ Copy tests passed" << std::endl;
}

int main() {
    std::cout << "\n=== Tensor Test Suite ===" << std::endl;

    test_tensor_creation();
    test_tensor_indexing();
    test_tensor_reshape();
    test_tensor_operations();
    test_tensor_transpose();
    test_tensor_mat_interop();
    test_tensor_squeeze_unsqueeze();
    test_tensor_statistics();
    test_tensor_copy();

    std::cout << "\n✓ All tensor tests passed!" << std::endl;
    return 0;
}
