#pragma once

#include "Matrix/matrix.h"
#include <cmath>
#include <algorithm>

namespace ml {

    // Activation function types
    enum class ActivationType {
        SIGMOID,
        RELU,
        LEAKY_RELU,
        TANH,
        SOFTMAX,
        ELU,
        SELU,
        LINEAR
    };

    // Convert activation type to string for debugging/logging
    inline const char* ActivationTypeToString(ActivationType type) {
        switch (type) {
            case ActivationType::SIGMOID: return "Sigmoid";
            case ActivationType::RELU: return "ReLU";
            case ActivationType::LEAKY_RELU: return "LeakyReLU";
            case ActivationType::TANH: return "Tanh";
            case ActivationType::SOFTMAX: return "Softmax";
            case ActivationType::ELU: return "ELU";
            case ActivationType::SELU: return "SELU";
            case ActivationType::LINEAR: return "Linear";
            default: return "Unknown";
        }
    }

    // ========== ACTIVATION FUNCTIONS ==========

    // Sigmoid: σ(x) = 1 / (1 + e^(-x))
    // Range: (0, 1)
    // Use: Binary classification output, historically used in hidden layers
    template <typename T>
    ml::Mat<T> Sigmoid(const ml::Mat<T>& mat) {
        ml::Mat<T> result(mat.size(), 0);
        for (int i = 0; i < mat.size().cy; ++i) {
            for (int j = 0; j < mat.size().cx; ++j) {
                T val = mat.getAt(i, j);
                result.setAt(i, j, 1.0 / (1.0 + std::exp(-val)));
            }
        }
        return result;
    }

    // ReLU: f(x) = max(0, x)
    // Range: [0, ∞)
    // Use: Default choice for hidden layers in deep networks
    template <typename T>
    ml::Mat<T> ReLU(const ml::Mat<T>& mat) {
        ml::Mat<T> result(mat.size(), 0);
        for (int i = 0; i < mat.size().cy; ++i) {
            for (int j = 0; j < mat.size().cx; ++j) {
                T val = mat.getAt(i, j);
                result.setAt(i, j, std::max(T(0), val));
            }
        }
        return result;
    }

    // Leaky ReLU: f(x) = max(αx, x) where α is typically 0.01
    // Range: (-∞, ∞)
    // Use: Prevents dying ReLU problem by allowing small negative values
    template <typename T>
    ml::Mat<T> LeakyReLU(const ml::Mat<T>& mat, T alpha = 0.01) {
        ml::Mat<T> result(mat.size(), 0);
        for (int i = 0; i < mat.size().cy; ++i) {
            for (int j = 0; j < mat.size().cx; ++j) {
                T val = mat.getAt(i, j);
                result.setAt(i, j, val > 0 ? val : alpha * val);
            }
        }
        return result;
    }

    // Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    // Range: (-1, 1)
    // Use: Hidden layers, zero-centered (unlike sigmoid)
    template <typename T>
    ml::Mat<T> Tanh(const ml::Mat<T>& mat) {
        ml::Mat<T> result(mat.size(), 0);
        for (int i = 0; i < mat.size().cy; ++i) {
            for (int j = 0; j < mat.size().cx; ++j) {
                T val = mat.getAt(i, j);
                result.setAt(i, j, std::tanh(val));
            }
        }
        return result;
    }

    // Softmax: f(x_i) = e^(x_i) / Σ(e^(x_j))
    // Range: (0, 1), outputs sum to 1
    // Use: Multi-class classification output layer
    template <typename T>
    ml::Mat<T> Softmax(const ml::Mat<T>& mat) {
        ml::Mat<T> result(mat.size(), 0);

        // Process each row independently (each row is a sample)
        for (int i = 0; i < mat.size().cy; ++i) {
            // Find max for numerical stability
            T maxVal = mat.getAt(i, 0);
            for (int j = 1; j < mat.size().cx; ++j) {
                maxVal = std::max(maxVal, mat.getAt(i, j));
            }

            // Compute exp(x - max) and sum
            T sum = 0;
            for (int j = 0; j < mat.size().cx; ++j) {
                T expVal = std::exp(mat.getAt(i, j) - maxVal);
                result.setAt(i, j, expVal);
                sum += expVal;
            }

            // Normalize
            for (int j = 0; j < mat.size().cx; ++j) {
                result.setAt(i, j, result.getAt(i, j) / sum);
            }
        }
        return result;
    }

    // ELU: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
    // Range: (-α, ∞) where α is typically 1.0
    // Use: Can produce negative outputs, smoother than ReLU
    template <typename T>
    ml::Mat<T> ELU(const ml::Mat<T>& mat, T alpha = 1.0) {
        ml::Mat<T> result(mat.size(), 0);
        for (int i = 0; i < mat.size().cy; ++i) {
            for (int j = 0; j < mat.size().cx; ++j) {
                T val = mat.getAt(i, j);
                result.setAt(i, j, val > 0 ? val : alpha * (std::exp(val) - 1));
            }
        }
        return result;
    }

    // SELU: Self-normalizing variant of ELU with specific constants
    // λ ≈ 1.0507, α ≈ 1.67326
    // Range: (-λα, ∞)
    // Use: Self-normalizing properties for deep networks
    template <typename T>
    ml::Mat<T> SELU(const ml::Mat<T>& mat) {
        const T lambda = 1.0507009873554804934193349852946;
        const T alpha = 1.6732632423543772848170429916717;

        ml::Mat<T> result(mat.size(), 0);
        for (int i = 0; i < mat.size().cy; ++i) {
            for (int j = 0; j < mat.size().cx; ++j) {
                T val = mat.getAt(i, j);
                result.setAt(i, j, lambda * (val > 0 ? val : alpha * (std::exp(val) - 1)));
            }
        }
        return result;
    }

    // Linear: f(x) = x
    // Range: (-∞, ∞)
    // Use: Regression output layer
    template <typename T>
    ml::Mat<T> Linear(const ml::Mat<T>& mat) {
        return mat.Copy();
    }

    // ========== ACTIVATION GRADIENTS ==========

    // Sigmoid gradient: σ'(x) = σ(x) * (1 - σ(x))
    // Note: Input should be the ACTIVATED values, not raw inputs
    template <typename T>
    ml::Mat<T> SigmoidGrad(const ml::Mat<T>& activated) {
        return ml::ElementMult(activated, ml::Diff<T>(1, activated));
    }

    // ReLU gradient: f'(x) = 1 if x > 0, 0 otherwise
    // Note: Input should be the ACTIVATED values
    template <typename T>
    ml::Mat<T> ReLUGrad(const ml::Mat<T>& activated) {
        ml::Mat<T> result(activated.size(), 0);
        for (int i = 0; i < activated.size().cy; ++i) {
            for (int j = 0; j < activated.size().cx; ++j) {
                T val = activated.getAt(i, j);
                result.setAt(i, j, val > 0 ? T(1) : T(0));
            }
        }
        return result;
    }

    // Leaky ReLU gradient: f'(x) = 1 if x > 0, α otherwise
    template <typename T>
    ml::Mat<T> LeakyReLUGrad(const ml::Mat<T>& activated, T alpha = 0.01) {
        ml::Mat<T> result(activated.size(), 0);
        for (int i = 0; i < activated.size().cy; ++i) {
            for (int j = 0; j < activated.size().cx; ++j) {
                T val = activated.getAt(i, j);
                result.setAt(i, j, val > 0 ? T(1) : alpha);
            }
        }
        return result;
    }

    // Tanh gradient: f'(x) = 1 - tanh²(x)
    // Note: Input should be the ACTIVATED values
    template <typename T>
    ml::Mat<T> TanhGrad(const ml::Mat<T>& activated) {
        ml::Mat<T> result(activated.size(), 0);
        for (int i = 0; i < activated.size().cy; ++i) {
            for (int j = 0; j < activated.size().cx; ++j) {
                T val = activated.getAt(i, j);
                result.setAt(i, j, 1 - val * val);
            }
        }
        return result;
    }

    // Softmax gradient: For cross-entropy loss, gradient simplifies to (predicted - target)
    // This is handled in the loss function, so we return 1s
    // For general case: ∂y_i/∂x_j = y_i(δ_ij - y_j) where δ is Kronecker delta
    template <typename T>
    ml::Mat<T> SoftmaxGrad(const ml::Mat<T>& activated) {
        // When used with cross-entropy loss, the gradient simplifies
        // and is typically computed differently in the loss function
        // For now, return identity (will be properly handled in loss computation)
        ml::Mat<T> result(activated.size(), 0);
        for (int i = 0; i < activated.size().cy; ++i) {
            for (int j = 0; j < activated.size().cx; ++j) {
                result.setAt(i, j, T(1));
            }
        }
        return result;
    }

    // ELU gradient: f'(x) = 1 if x > 0, α*e^x if x ≤ 0
    // Note: For activated values, if val > 0 then gradient is 1
    // If val ≤ 0, then val = α(e^x - 1), so e^x = val/α + 1
    template <typename T>
    ml::Mat<T> ELUGrad(const ml::Mat<T>& activated, T alpha = 1.0) {
        ml::Mat<T> result(activated.size(), 0);
        for (int i = 0; i < activated.size().cy; ++i) {
            for (int j = 0; j < activated.size().cx; ++j) {
                T val = activated.getAt(i, j);
                if (val > 0) {
                    result.setAt(i, j, T(1));
                } else {
                    // val = α(e^x - 1), so gradient = α*e^x = val + α
                    result.setAt(i, j, val + alpha);
                }
            }
        }
        return result;
    }

    // SELU gradient: Similar to ELU but with SELU constants
    template <typename T>
    ml::Mat<T> SELUGrad(const ml::Mat<T>& activated) {
        const T lambda = 1.0507009873554804934193349852946;
        const T alpha = 1.6732632423543772848170429916717;

        ml::Mat<T> result(activated.size(), 0);
        for (int i = 0; i < activated.size().cy; ++i) {
            for (int j = 0; j < activated.size().cx; ++j) {
                T val = activated.getAt(i, j);
                if (val > 0) {
                    result.setAt(i, j, lambda);
                } else {
                    // val = λ*α(e^x - 1), gradient = λ*α*e^x = val + λ*α
                    result.setAt(i, j, val + lambda * alpha);
                }
            }
        }
        return result;
    }

    // Linear gradient: f'(x) = 1
    template <typename T>
    ml::Mat<T> LinearGrad(const ml::Mat<T>& activated) {
        ml::Mat<T> result(activated.size(), 0);
        for (int i = 0; i < activated.size().cy; ++i) {
            for (int j = 0; j < activated.size().cx; ++j) {
                result.setAt(i, j, T(1));
            }
        }
        return result;
    }

    // ========== UNIFIED INTERFACE ==========

    // Apply activation function based on type
    template <typename T>
    ml::Mat<T> Activate(const ml::Mat<T>& input, ActivationType type, T alpha = 0.01) {
        switch (type) {
            case ActivationType::SIGMOID:
                return Sigmoid(input);
            case ActivationType::RELU:
                return ReLU(input);
            case ActivationType::LEAKY_RELU:
                return LeakyReLU(input, alpha);
            case ActivationType::TANH:
                return Tanh(input);
            case ActivationType::SOFTMAX:
                return Softmax(input);
            case ActivationType::ELU:
                return ELU(input, alpha);
            case ActivationType::SELU:
                return SELU(input);
            case ActivationType::LINEAR:
                return Linear(input);
            default:
                return Sigmoid(input); // Default fallback
        }
    }

    // Apply activation gradient based on type
    template <typename T>
    ml::Mat<T> ActivateGrad(const ml::Mat<T>& activated, ActivationType type, T alpha = 0.01) {
        switch (type) {
            case ActivationType::SIGMOID:
                return SigmoidGrad(activated);
            case ActivationType::RELU:
                return ReLUGrad(activated);
            case ActivationType::LEAKY_RELU:
                return LeakyReLUGrad(activated, alpha);
            case ActivationType::TANH:
                return TanhGrad(activated);
            case ActivationType::SOFTMAX:
                return SoftmaxGrad(activated);
            case ActivationType::ELU:
                return ELUGrad(activated, alpha);
            case ActivationType::SELU:
                return SELUGrad(activated);
            case ActivationType::LINEAR:
                return LinearGrad(activated);
            default:
                return SigmoidGrad(activated); // Default fallback
        }
    }

} // namespace ml
