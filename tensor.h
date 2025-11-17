#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <random>
#include <cmath>
#include <iostream>
#include "math/rect.h"
#include "Matrix/matrix.h"

/**
 * N-dimensional tensor class for CNN support
 *
 * Design principles:
 * - Row-major (C-style) memory layout
 * - Shared pointer semantics (cheap copying)
 * - Interoperable with Mat<T>
 * - Supports arbitrary dimensions
 *
 * Common shapes:
 * - 2D: [height, width]
 * - 3D: [channels, height, width] or [batch, height, width]
 * - 4D: [batch, channels, height, width]
 */
template <typename T>
class Tensor {
private:
    std::vector<size_t> mShape;
    std::shared_ptr<T[]> mData;
    size_t mSize;  // Total number of elements

    /**
     * Compute strides for row-major indexing
     * Example: shape [2, 3, 4] → strides [12, 4, 1]
     */
    std::vector<size_t> computeStrides() const {
        std::vector<size_t> strides(mShape.size(), 1);
        for (int i = (int)mShape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * mShape[i + 1];
        }
        return strides;
    }

    /**
     * Convert multi-dimensional index to linear index
     */
    size_t indexToOffset(const std::vector<size_t>& indices) const {
        if (indices.size() != mShape.size()) {
            throw std::runtime_error("Index dimension mismatch");
        }

        auto strides = computeStrides();
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= mShape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            offset += indices[i] * strides[i];
        }
        return offset;
    }

public:
    // ========== Constructors ==========

    /**
     * Default constructor - empty tensor
     */
    Tensor() : mShape{}, mData(nullptr), mSize(0) {}

    /**
     * Create tensor with given shape (uninitialized)
     */
    explicit Tensor(const std::vector<size_t>& shape) : mShape(shape) {
        mSize = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
        if (mSize == 0) {
            throw std::runtime_error("Cannot create tensor with zero size");
        }
        mData = std::shared_ptr<T[]>(new T[mSize]);
    }

    /**
     * Create tensor with shape and fill value
     */
    Tensor(const std::vector<size_t>& shape, T fillValue) : Tensor(shape) {
        std::fill(mData.get(), mData.get() + mSize, fillValue);
    }

    /**
     * Create tensor from raw data (copies data)
     */
    Tensor(const std::vector<size_t>& shape, const T* data) : Tensor(shape) {
        std::memcpy(mData.get(), data, mSize * sizeof(T));
    }

    /**
     * Create tensor from std::vector data
     */
    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data) : Tensor(shape) {
        if (data.size() != mSize) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        std::memcpy(mData.get(), data.data(), mSize * sizeof(T));
    }

    // ========== Factory Methods ==========

    /**
     * Create tensor filled with zeros
     */
    static Tensor<T> zeros(const std::vector<size_t>& shape) {
        return Tensor<T>(shape, T(0));
    }

    /**
     * Create tensor filled with ones
     */
    static Tensor<T> ones(const std::vector<size_t>& shape) {
        return Tensor<T>(shape, T(1));
    }

    /**
     * Create tensor with random values (uniform [0, 1])
     */
    static Tensor<T> random(const std::vector<size_t>& shape, T min = T(0), T max = T(1)) {
        Tensor<T> tensor(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min, max);

        for (size_t i = 0; i < tensor.mSize; ++i) {
            tensor.mData[i] = dist(gen);
        }
        return tensor;
    }

    /**
     * Create tensor with random normal distribution
     */
    static Tensor<T> randn(const std::vector<size_t>& shape, T mean = T(0), T stddev = T(1)) {
        Tensor<T> tensor(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(mean, stddev);

        for (size_t i = 0; i < tensor.mSize; ++i) {
            tensor.mData[i] = dist(gen);
        }
        return tensor;
    }

    /**
     * Create tensor from ml::Mat<T> with new shape
     * Useful for: Dense layer output → Conv layer input
     */
    static Tensor<T> fromMat(const ml::Mat<T>& mat, const std::vector<size_t>& shape) {
        auto matSize = mat.size();
        size_t matRows = matSize.cy;   // cy is height (row count)
        size_t matCols = matSize.cx;   // cx is width (col count)
        size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
        if (totalSize != matRows * matCols) {
            throw std::runtime_error("Mat size does not match target tensor size");
        }

        Tensor<T> tensor(shape);
        // Copy data from mat (assuming row-major layout)
        for (size_t i = 0; i < matRows; ++i) {
            for (size_t j = 0; j < matCols; ++j) {
                tensor.mData[i * matCols + j] = mat.getAt(i, j);
            }
        }
        return tensor;
    }

    // ========== Properties ==========

    const std::vector<size_t>& shape() const { return mShape; }
    size_t ndim() const { return mShape.size(); }
    size_t size() const { return mSize; }
    T* data() { return mData.get(); }
    const T* data() const { return mData.get(); }

    /**
     * Get specific dimension size
     */
    size_t shape(size_t dim) const {
        if (dim >= mShape.size()) {
            throw std::out_of_range("Dimension index out of range");
        }
        return mShape[dim];
    }

    // ========== Indexing ==========

    /**
     * 1D indexing (linear)
     */
    T& operator()(size_t i) {
        if (i >= mSize) throw std::out_of_range("Index out of bounds");
        return mData[i];
    }

    const T& operator()(size_t i) const {
        if (i >= mSize) throw std::out_of_range("Index out of bounds");
        return mData[i];
    }

    /**
     * 2D indexing
     */
    T& operator()(size_t i, size_t j) {
        if (mShape.size() != 2) throw std::runtime_error("2D indexing requires 2D tensor");
        return mData[i * mShape[1] + j];
    }

    const T& operator()(size_t i, size_t j) const {
        if (mShape.size() != 2) throw std::runtime_error("2D indexing requires 2D tensor");
        return mData[i * mShape[1] + j];
    }

    /**
     * 3D indexing
     */
    T& operator()(size_t i, size_t j, size_t k) {
        if (mShape.size() != 3) throw std::runtime_error("3D indexing requires 3D tensor");
        return mData[i * mShape[1] * mShape[2] + j * mShape[2] + k];
    }

    const T& operator()(size_t i, size_t j, size_t k) const {
        if (mShape.size() != 3) throw std::runtime_error("3D indexing requires 3D tensor");
        return mData[i * mShape[1] * mShape[2] + j * mShape[2] + k];
    }

    /**
     * 4D indexing (batch, channels, height, width)
     */
    T& operator()(size_t i, size_t j, size_t k, size_t l) {
        if (mShape.size() != 4) throw std::runtime_error("4D indexing requires 4D tensor");
        return mData[i * mShape[1] * mShape[2] * mShape[3] +
                     j * mShape[2] * mShape[3] +
                     k * mShape[3] +
                     l];
    }

    const T& operator()(size_t i, size_t j, size_t k, size_t l) const {
        if (mShape.size() != 4) throw std::runtime_error("4D indexing requires 4D tensor");
        return mData[i * mShape[1] * mShape[2] * mShape[3] +
                     j * mShape[2] * mShape[3] +
                     k * mShape[3] +
                     l];
    }

    /**
     * General N-D indexing
     */
    T& at(const std::vector<size_t>& indices) {
        return mData[indexToOffset(indices)];
    }

    const T& at(const std::vector<size_t>& indices) const {
        return mData[indexToOffset(indices)];
    }

    // ========== Shape Manipulation ==========

    /**
     * Reshape tensor (must preserve total size)
     */
    Tensor<T> reshape(const std::vector<size_t>& newShape) const {
        size_t newSize = std::accumulate(newShape.begin(), newShape.end(), 1ULL, std::multiplies<size_t>());
        if (newSize != mSize) {
            std::ostringstream oss;
            oss << "Cannot reshape tensor of size " << mSize << " to size " << newSize;
            throw std::runtime_error(oss.str());
        }

        Tensor<T> result(newShape);
        std::memcpy(result.mData.get(), mData.get(), mSize * sizeof(T));
        return result;
    }

    /**
     * Flatten to 1D tensor
     */
    Tensor<T> flatten() const {
        return reshape({mSize});
    }

    /**
     * Flatten to 2D ml::Mat (for compatibility)
     */
    ml::Mat<T> toMat() const {
        if (mShape.size() == 2) {
            // Already 2D, direct conversion
            ml::Mat<T> mat(mShape[0], mShape[1]);
            for (size_t i = 0; i < mShape[0]; ++i) {
                for (size_t j = 0; j < mShape[1]; ++j) {
                    mat.setAt(i, j, (*this)(i, j));
                }
            }
            return mat;
        } else {
            // Flatten to [1, size]
            ml::Mat<T> mat(1, mSize);
            for (size_t i = 0; i < mSize; ++i) {
                mat.setAt(0, i, mData[i]);
            }
            return mat;
        }
    }

    /**
     * Add new axis at specified position
     * Example: [3, 4] with axis=0 → [1, 3, 4]
     */
    Tensor<T> unsqueeze(int axis) const {
        if (axis < 0) axis += (int)mShape.size() + 1;
        if (axis < 0 || axis > (int)mShape.size()) {
            throw std::out_of_range("Axis out of range");
        }

        std::vector<size_t> newShape = mShape;
        newShape.insert(newShape.begin() + axis, 1);
        return reshape(newShape);
    }

    /**
     * Remove axes of size 1
     * Example: [1, 3, 1, 4] → [3, 4]
     */
    Tensor<T> squeeze() const {
        std::vector<size_t> newShape;
        for (size_t dim : mShape) {
            if (dim != 1) {
                newShape.push_back(dim);
            }
        }
        if (newShape.empty()) {
            newShape.push_back(1);  // Scalar → [1]
        }
        return reshape(newShape);
    }

    // ========== Element-wise Operations ==========

    /**
     * Element-wise addition
     */
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (mShape != other.mShape) {
            throw std::runtime_error("Shape mismatch for addition");
        }

        Tensor<T> result(mShape);
        for (size_t i = 0; i < mSize; ++i) {
            result.mData[i] = mData[i] + other.mData[i];
        }
        return result;
    }

    /**
     * Element-wise subtraction
     */
    Tensor<T> operator-(const Tensor<T>& other) const {
        if (mShape != other.mShape) {
            throw std::runtime_error("Shape mismatch for subtraction");
        }

        Tensor<T> result(mShape);
        for (size_t i = 0; i < mSize; ++i) {
            result.mData[i] = mData[i] - other.mData[i];
        }
        return result;
    }

    /**
     * Element-wise multiplication (Hadamard product)
     */
    Tensor<T> operator*(const Tensor<T>& other) const {
        if (mShape != other.mShape) {
            throw std::runtime_error("Shape mismatch for element-wise multiplication");
        }

        Tensor<T> result(mShape);
        for (size_t i = 0; i < mSize; ++i) {
            result.mData[i] = mData[i] * other.mData[i];
        }
        return result;
    }

    /**
     * Scalar multiplication
     */
    Tensor<T> operator*(T scalar) const {
        Tensor<T> result(mShape);
        for (size_t i = 0; i < mSize; ++i) {
            result.mData[i] = mData[i] * scalar;
        }
        return result;
    }

    /**
     * Scalar division
     */
    Tensor<T> operator/(T scalar) const {
        if (scalar == T(0)) {
            throw std::runtime_error("Division by zero");
        }

        Tensor<T> result(mShape);
        for (size_t i = 0; i < mSize; ++i) {
            result.mData[i] = mData[i] / scalar;
        }
        return result;
    }

    /**
     * In-place addition
     */
    Tensor<T>& operator+=(const Tensor<T>& other) {
        if (mShape != other.mShape) {
            throw std::runtime_error("Shape mismatch for addition");
        }

        for (size_t i = 0; i < mSize; ++i) {
            mData[i] += other.mData[i];
        }
        return *this;
    }

    /**
     * In-place scalar multiplication
     */
    Tensor<T>& operator*=(T scalar) {
        for (size_t i = 0; i < mSize; ++i) {
            mData[i] *= scalar;
        }
        return *this;
    }

    // ========== Utility Functions ==========

    /**
     * Copy tensor (deep copy)
     */
    Tensor<T> copy() const {
        Tensor<T> result(mShape);
        std::memcpy(result.mData.get(), mData.get(), mSize * sizeof(T));
        return result;
    }

    /**
     * Fill with value
     */
    void fill(T value) {
        std::fill(mData.get(), mData.get() + mSize, value);
    }

    /**
     * Get sum of all elements
     */
    T sum() const {
        T result = T(0);
        for (size_t i = 0; i < mSize; ++i) {
            result += mData[i];
        }
        return result;
    }

    /**
     * Get mean of all elements
     */
    T mean() const {
        return sum() / T(mSize);
    }

    /**
     * Get max element
     */
    T max() const {
        if (mSize == 0) throw std::runtime_error("Cannot get max of empty tensor");
        T maxVal = mData[0];
        for (size_t i = 1; i < mSize; ++i) {
            if (mData[i] > maxVal) maxVal = mData[i];
        }
        return maxVal;
    }

    /**
     * Get min element
     */
    T min() const {
        if (mSize == 0) throw std::runtime_error("Cannot get min of empty tensor");
        T minVal = mData[0];
        for (size_t i = 1; i < mSize; ++i) {
            if (mData[i] < minVal) minVal = mData[i];
        }
        return minVal;
    }

    /**
     * Print tensor shape and some values (for debugging)
     */
    void print(const std::string& name = "Tensor") const {
        std::cout << name << " shape: [";
        for (size_t i = 0; i < mShape.size(); ++i) {
            std::cout << mShape[i];
            if (i < mShape.size() - 1) std::cout << ", ";
        }
        std::cout << "]";

        std::cout << " size: " << mSize;
        std::cout << " range: [" << min() << ", " << max() << "]";
        std::cout << " mean: " << mean() << std::endl;

        // Print first few values
        std::cout << "  First values: ";
        size_t printCount = std::min(size_t(10), mSize);
        for (size_t i = 0; i < printCount; ++i) {
            std::cout << mData[i] << " ";
        }
        if (mSize > printCount) std::cout << "...";
        std::cout << std::endl;
    }

    /**
     * Transpose for 2D tensors (matrix transpose)
     */
    Tensor<T> transpose() const {
        if (mShape.size() != 2) {
            throw std::runtime_error("Simple transpose only works for 2D tensors");
        }

        std::vector<size_t> newShape = {mShape[1], mShape[0]};
        Tensor<T> result(newShape);

        for (size_t i = 0; i < mShape[0]; ++i) {
            for (size_t j = 0; j < mShape[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    /**
     * Transpose with arbitrary axis permutation
     * Example: [B, C, H, W] with axes=[0, 2, 3, 1] → [B, H, W, C]
     */
    Tensor<T> transpose(const std::vector<int>& axes) const {
        if (axes.size() != mShape.size()) {
            throw std::runtime_error("Axes size must match tensor dimensions");
        }

        // Build new shape
        std::vector<size_t> newShape(mShape.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            int axis = axes[i];
            if (axis < 0) axis += (int)mShape.size();
            if (axis < 0 || axis >= (int)mShape.size()) {
                throw std::out_of_range("Axis out of range");
            }
            newShape[i] = mShape[axis];
        }

        Tensor<T> result(newShape);

        // Transpose data (slow implementation, can optimize later)
        std::vector<size_t> indices(mShape.size(), 0);
        for (size_t linearIdx = 0; linearIdx < mSize; ++linearIdx) {
            // Convert linear index to multi-index in original tensor
            size_t temp = linearIdx;
            auto strides = computeStrides();
            for (size_t i = 0; i < mShape.size(); ++i) {
                indices[i] = temp / strides[i];
                temp %= strides[i];
            }

            // Permute indices according to axes
            std::vector<size_t> newIndices(indices.size());
            for (size_t i = 0; i < axes.size(); ++i) {
                int axis = axes[i];
                if (axis < 0) axis += (int)mShape.size();
                newIndices[i] = indices[axis];
            }

            // Set value in result tensor
            result.at(newIndices) = mData[linearIdx];
        }

        return result;
    }
};

// ========== Non-member Functions ==========

/**
 * Scalar * Tensor (commutative)
 */
template <typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& tensor) {
    return tensor * scalar;
}

#endif // TENSOR_H
