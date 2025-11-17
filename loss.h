#pragma once

#include <cmath>
#include <algorithm>
#include "Matrix/matrix.h"

namespace ml {

/**
 * Loss function types
 */
enum class LossType {
    MSE,                    // Mean Squared Error (for regression)
    CROSS_ENTROPY,          // Cross-Entropy (for classification)
    BINARY_CROSS_ENTROPY    // Binary Cross-Entropy (for binary classification)
};

/**
 * Mean Squared Error (MSE) Loss
 * L = (1/2) * sum((target - predicted)^2)
 * Gradient: dL/dy = -(target - predicted) = (predicted - target)
 */
template <typename T>
ml::Mat<T> MSELoss(const ml::Mat<T>& predicted, const ml::Mat<T>& target) {
    return ml::Diff(predicted, target);
}

/**
 * Cross-Entropy Loss for multi-class classification
 * L = -sum(target * log(predicted))
 *
 * For softmax output with one-hot encoded targets:
 * Gradient: dL/dy = predicted - target
 *
 * For sigmoid output (treating each class independently):
 * Gradient: dL/dy = -(target / predicted) + (1 - target) / (1 - predicted)
 * But when combined with sigmoid, simplifies to: predicted - target
 *
 * @param predicted Network output (should be probabilities)
 * @param target One-hot encoded target labels
 * @return Error gradient for backpropagation
 */
template <typename T>
ml::Mat<T> CrossEntropyLoss(const ml::Mat<T>& predicted, const ml::Mat<T>& target) {
    // The gradient of cross-entropy loss with respect to pre-activation (logits)
    // when using softmax/sigmoid activation is simply: predicted - target
    // This is a beautiful property that makes training stable!
    return ml::Diff(predicted, target);
}

/**
 * Binary Cross-Entropy Loss (for binary classification)
 * L = -[target * log(predicted) + (1 - target) * log(1 - predicted)]
 * Gradient: dL/dy = predicted - target (when combined with sigmoid)
 */
template <typename T>
ml::Mat<T> BinaryCrossEntropyLoss(const ml::Mat<T>& predicted, const ml::Mat<T>& target) {
    return ml::Diff(predicted, target);
}

/**
 * Compute loss value (for monitoring training progress)
 * Returns the scalar loss value
 */
template <typename T>
T ComputeLoss(const ml::Mat<T>& predicted, const ml::Mat<T>& target, LossType lossType) {
    T totalLoss = 0.0;
    int numSamples = predicted.size().cy;
    int numOutputs = predicted.size().cx;

    const T epsilon = 1e-7; // Small constant to avoid log(0)

    switch (lossType) {
        case LossType::MSE: {
            // Mean Squared Error: (1/2n) * sum((predicted - target)^2)
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numOutputs; j++) {
                    T diff = predicted.getAt(i, j) - target.getAt(i, j);
                    totalLoss += diff * diff;
                }
            }
            return totalLoss / (2.0 * numSamples);
        }

        case LossType::CROSS_ENTROPY: {
            // Cross-Entropy: -(1/n) * sum(target * log(predicted))
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numOutputs; j++) {
                    T pred = std::max(epsilon, std::min(T(1.0) - epsilon, predicted.getAt(i, j)));
                    T targ = target.getAt(i, j);
                    if (targ > 0) {  // Only compute for non-zero targets (one-hot encoding)
                        totalLoss += -targ * std::log(pred);
                    }
                }
            }
            return totalLoss / numSamples;
        }

        case LossType::BINARY_CROSS_ENTROPY: {
            // Binary Cross-Entropy: -(1/n) * sum[target*log(pred) + (1-target)*log(1-pred)]
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numOutputs; j++) {
                    T pred = std::max(epsilon, std::min(T(1.0) - epsilon, predicted.getAt(i, j)));
                    T targ = target.getAt(i, j);
                    totalLoss += -(targ * std::log(pred) + (T(1.0) - targ) * std::log(T(1.0) - pred));
                }
            }
            return totalLoss / numSamples;
        }

        default:
            return totalLoss;
    }
}

/**
 * Compute loss gradient for backpropagation
 * Returns the error gradient: dL/dy
 */
template <typename T>
ml::Mat<T> ComputeLossGradient(const ml::Mat<T>& predicted, const ml::Mat<T>& target,
                                LossType lossType) {
    switch (lossType) {
        case LossType::MSE:
            return MSELoss(predicted, target);

        case LossType::CROSS_ENTROPY:
            return CrossEntropyLoss(predicted, target);

        case LossType::BINARY_CROSS_ENTROPY:
            return BinaryCrossEntropyLoss(predicted, target);

        default:
            return MSELoss(predicted, target);
    }
}

/**
 * Compute classification accuracy
 * For multi-class classification with one-hot encoded labels
 *
 * @param predicted Network output probabilities (batch_size, num_classes)
 * @param target One-hot encoded targets (batch_size, num_classes)
 * @return Accuracy as a percentage (0-100)
 */
template <typename T>
T ComputeAccuracy(const ml::Mat<T>& predicted, const ml::Mat<T>& target) {
    int numSamples = predicted.size().cy;
    int numClasses = predicted.size().cx;
    int correct = 0;

    for (int i = 0; i < numSamples; i++) {
        // Find predicted class (argmax of predictions)
        int predictedClass = 0;
        T maxPred = predicted.getAt(i, 0);
        for (int j = 1; j < numClasses; j++) {
            T pred = predicted.getAt(i, j);
            if (pred > maxPred) {
                maxPred = pred;
                predictedClass = j;
            }
        }

        // Find true class (argmax of one-hot target)
        int trueClass = 0;
        T maxTarget = target.getAt(i, 0);
        for (int j = 1; j < numClasses; j++) {
            T targ = target.getAt(i, j);
            if (targ > maxTarget) {
                maxTarget = targ;
                trueClass = j;
            }
        }

        if (predictedClass == trueClass) {
            correct++;
        }
    }

    return (T(100.0) * correct) / numSamples;
}

} // namespace ml
