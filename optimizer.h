#pragma once

#include "Matrix/matrix.h"
#include <map>
#include <string>
#include <cmath>

namespace ml {

    // Optimizer types
    enum class OptimizerType {
        SGD,
        MOMENTUM,
        ADAM
    };

    // Convert optimizer type to string for debugging/logging
    inline const char* OptimizerTypeToString(OptimizerType type) {
        switch (type) {
            case OptimizerType::SGD: return "SGD";
            case OptimizerType::MOMENTUM: return "Momentum";
            case OptimizerType::ADAM: return "Adam";
            default: return "Unknown";
        }
    }

    // Base optimizer interface
    template <typename T>
    class IOptimizer {
    public:
        virtual ~IOptimizer() {}

        // Update weights given gradients
        // layerKey is used to track state for each layer-sibling pair
        virtual void updateWeights(
            ml::Mat<T>& weights,
            const ml::Mat<T>& gradients,
            T learningRate,
            const std::string& layerKey
        ) = 0;

        // Reset optimizer state (useful when starting new training)
        virtual void reset() = 0;

        // Get optimizer type
        virtual OptimizerType getType() const = 0;
    };

    // ========== SGD OPTIMIZER ==========
    // Stochastic Gradient Descent with optional gradient clipping
    // Update: W = W + learningRate * gradient
    template <typename T>
    class SGDOptimizer : public IOptimizer<T> {
    public:
        SGDOptimizer(T gradientClipThreshold = 10.0)
            : mGradientClipThreshold(gradientClipThreshold) {}

        virtual ~SGDOptimizer() {}

        virtual void updateWeights(
            ml::Mat<T>& weights,
            const ml::Mat<T>& gradients,
            T learningRate,
            const std::string& layerKey
        ) override {
            if (!weights.IsGood() || !gradients.IsGood()) return;
            if (weights.size() != gradients.size()) return;

            for (int row = 0; row < weights.size().cy; ++row) {
                for (int col = 0; col < weights.size().cx; ++col) {
                    T gradient = gradients.getAt(row, col);

                    // Clip gradient to prevent exploding gradients
                    if (std::abs(gradient) > mGradientClipThreshold) {
                        gradient = (gradient > 0) ? mGradientClipThreshold : -mGradientClipThreshold;
                    }

                    T currentWeight = weights.getAt(row, col);
                    T newWeight = currentWeight + learningRate * gradient;
                    weights.setAt(row, col, newWeight);
                }
            }
        }

        virtual void reset() override {
            // SGD has no state to reset
        }

        virtual OptimizerType getType() const override {
            return OptimizerType::SGD;
        }

    private:
        T mGradientClipThreshold;
    };

    // ========== MOMENTUM OPTIMIZER ==========
    // SGD with Momentum: accumulates velocity vector
    // velocity = β * velocity + gradient
    // W = W + learningRate * velocity
    //
    // Hyperparameters:
    // - β (beta): momentum coefficient, typically 0.9
    // - Helps accelerate in relevant direction and dampen oscillations
    template <typename T>
    class MomentumOptimizer : public IOptimizer<T> {
    public:
        MomentumOptimizer(T beta = 0.9, T gradientClipThreshold = 10.0)
            : mBeta(beta), mGradientClipThreshold(gradientClipThreshold) {}

        virtual ~MomentumOptimizer() {}

        virtual void updateWeights(
            ml::Mat<T>& weights,
            const ml::Mat<T>& gradients,
            T learningRate,
            const std::string& layerKey
        ) override {
            if (!weights.IsGood() || !gradients.IsGood()) return;
            if (weights.size() != gradients.size()) return;

            // Initialize velocity for this layer if not exists
            if (mVelocity.find(layerKey) == mVelocity.end()) {
                mVelocity[layerKey] = ml::Mat<T>(weights.size(), 0);
            }

            ml::Mat<T>& velocity = mVelocity[layerKey];

            // Update velocity and weights
            for (int row = 0; row < weights.size().cy; ++row) {
                for (int col = 0; col < weights.size().cx; ++col) {
                    T gradient = gradients.getAt(row, col);

                    // Clip gradient
                    if (std::abs(gradient) > mGradientClipThreshold) {
                        gradient = (gradient > 0) ? mGradientClipThreshold : -mGradientClipThreshold;
                    }

                    // Update velocity: v = β*v + gradient
                    T v = mBeta * velocity.getAt(row, col) + gradient;
                    velocity.setAt(row, col, v);

                    // Update weights: W = W + learningRate * v
                    T currentWeight = weights.getAt(row, col);
                    T newWeight = currentWeight + learningRate * v;
                    weights.setAt(row, col, newWeight);
                }
            }
        }

        virtual void reset() override {
            mVelocity.clear();
        }

        virtual OptimizerType getType() const override {
            return OptimizerType::MOMENTUM;
        }

        T getBeta() const { return mBeta; }
        void setBeta(T beta) { mBeta = beta; }

    private:
        T mBeta;
        T mGradientClipThreshold;
        std::map<std::string, ml::Mat<T>> mVelocity; // velocity for each layer
    };

    // ========== ADAM OPTIMIZER ==========
    // Adaptive Moment Estimation
    // Combines momentum (first moment) with RMSprop (second moment)
    //
    // m = β1 * m + (1 - β1) * gradient        (first moment - mean)
    // v = β2 * v + (1 - β2) * gradient²       (second moment - variance)
    // m_hat = m / (1 - β1^t)                  (bias correction)
    // v_hat = v / (1 - β2^t)                  (bias correction)
    // W = W + learningRate * m_hat / (√v_hat + ε)
    //
    // Hyperparameters:
    // - β1: typically 0.9 (first moment decay)
    // - β2: typically 0.999 (second moment decay)
    // - ε: typically 1e-8 (small constant for numerical stability)
    template <typename T>
    class AdamOptimizer : public IOptimizer<T> {
    public:
        AdamOptimizer(T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8, T gradientClipThreshold = 10.0)
            : mBeta1(beta1), mBeta2(beta2), mEpsilon(epsilon),
              mGradientClipThreshold(gradientClipThreshold) {}

        virtual ~AdamOptimizer() {}

        virtual void updateWeights(
            ml::Mat<T>& weights,
            const ml::Mat<T>& gradients,
            T learningRate,
            const std::string& layerKey
        ) override {
            if (!weights.IsGood() || !gradients.IsGood()) return;
            if (weights.size() != gradients.size()) return;

            // Initialize moments for this layer if not exists
            if (mFirstMoment.find(layerKey) == mFirstMoment.end()) {
                mFirstMoment[layerKey] = ml::Mat<T>(weights.size(), 0);
                mSecondMoment[layerKey] = ml::Mat<T>(weights.size(), 0);
                mTimeStep[layerKey] = 0;
            }

            ml::Mat<T>& m = mFirstMoment[layerKey];
            ml::Mat<T>& v = mSecondMoment[layerKey];
            int& t = mTimeStep[layerKey];
            t++;

            // Compute bias correction terms
            T beta1_t = std::pow(mBeta1, t);
            T beta2_t = std::pow(mBeta2, t);
            T bias1Correction = 1.0 - beta1_t;
            T bias2Correction = 1.0 - beta2_t;

            // Update moments and weights
            for (int row = 0; row < weights.size().cy; ++row) {
                for (int col = 0; col < weights.size().cx; ++col) {
                    T gradient = gradients.getAt(row, col);

                    // Clip gradient
                    if (std::abs(gradient) > mGradientClipThreshold) {
                        gradient = (gradient > 0) ? mGradientClipThreshold : -mGradientClipThreshold;
                    }

                    // Update biased first moment estimate: m = β1*m + (1-β1)*g
                    T m_val = mBeta1 * m.getAt(row, col) + (1.0 - mBeta1) * gradient;
                    m.setAt(row, col, m_val);

                    // Update biased second moment estimate: v = β2*v + (1-β2)*g²
                    T v_val = mBeta2 * v.getAt(row, col) + (1.0 - mBeta2) * gradient * gradient;
                    v.setAt(row, col, v_val);

                    // Compute bias-corrected moment estimates
                    T m_hat = m_val / bias1Correction;
                    T v_hat = v_val / bias2Correction;

                    // Update weights: W = W + α * m_hat / (√v_hat + ε)
                    T currentWeight = weights.getAt(row, col);
                    T update = learningRate * m_hat / (std::sqrt(v_hat) + mEpsilon);
                    T newWeight = currentWeight + update;
                    weights.setAt(row, col, newWeight);
                }
            }
        }

        virtual void reset() override {
            mFirstMoment.clear();
            mSecondMoment.clear();
            mTimeStep.clear();
        }

        virtual OptimizerType getType() const override {
            return OptimizerType::ADAM;
        }

        T getBeta1() const { return mBeta1; }
        T getBeta2() const { return mBeta2; }
        T getEpsilon() const { return mEpsilon; }

        void setBeta1(T beta1) { mBeta1 = beta1; }
        void setBeta2(T beta2) { mBeta2 = beta2; }
        void setEpsilon(T epsilon) { mEpsilon = epsilon; }

    private:
        T mBeta1;
        T mBeta2;
        T mEpsilon;
        T mGradientClipThreshold;

        std::map<std::string, ml::Mat<T>> mFirstMoment;   // m - first moment (mean)
        std::map<std::string, ml::Mat<T>> mSecondMoment;  // v - second moment (variance)
        std::map<std::string, int> mTimeStep;             // t - timestep for bias correction
    };

    // ========== OPTIMIZER FACTORY ==========
    // Helper function to create optimizers
    template <typename T>
    IOptimizer<T>* CreateOptimizer(OptimizerType type) {
        switch (type) {
            case OptimizerType::SGD:
                return new SGDOptimizer<T>();
            case OptimizerType::MOMENTUM:
                return new MomentumOptimizer<T>();
            case OptimizerType::ADAM:
                return new AdamOptimizer<T>();
            default:
                return new SGDOptimizer<T>(); // Default fallback
        }
    }

} // namespace ml
