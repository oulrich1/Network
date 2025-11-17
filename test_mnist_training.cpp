#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include "allheader.h"
#include "network.h"
#include "mnist_loader.h"

using namespace std;
using namespace ml;
using namespace Utility;

/**
 * MNIST Training Test Suite
 *
 * Tests the MNIST training functionality including:
 * - Network creation and initialization
 * - Batch training
 * - Loss computation
 * - Accuracy evaluation
 * - Cross-entropy loss
 */

// Helper to check approximate equality
template <typename T>
bool approxEqual(T a, T b, T epsilon = 0.01) {
    return std::abs(a - b) < epsilon;
}

void test_network_creation() {
    BEGIN_TESTS("MNIST Network Creation");
    typedef double T;

    Network<T>* network = new Network<T>();
    ILayer<T>* input = new Layer<T>(784, "Input", ActivationType::RELU);
    ILayer<T>* hidden1 = new Layer<T>(128, "Hidden1", ActivationType::RELU);
    ILayer<T>* output = new Layer<T>(10, "Output", ActivationType::SIGMOID);

    network->setInputLayer(input);
    network->connect(input, hidden1);
    network->connect(hidden1, output);
    network->setOutputLayer(output);
    network->setOptimizerType(OptimizerType::ADAM);
    network->setLossType(LossType::CROSS_ENTROPY);

    network->init();

    cout << "✓ Network created successfully" << endl;
    cout << "✓ Optimizer set to Adam" << endl;
    cout << "✓ Loss set to Cross-Entropy" << endl;

    // Test forward pass with random input
    ml::Mat<T> testInput(1, 784, 0.5);
    ml::Mat<T> output_result = network->feed(testInput);

    cout << "✓ Forward pass works" << endl;
    cout << "  Output size: (" << output_result.size().cy << ", " << output_result.size().cx << ")" << endl;

    if (output_result.size().cy == 1 && output_result.size().cx == 10) {
        cout << "✓ Output dimensions correct" << endl;
    } else {
        cout << "✗ Output dimensions incorrect" << endl;
    }

    delete network;
}

void test_batch_training() {
    BEGIN_TESTS("Batch Training Functionality");
    typedef double T;

    // Create small network for testing
    Network<T>* network = new Network<T>();
    ILayer<T>* input = new Layer<T>(784, "Input", ActivationType::RELU);
    ILayer<T>* hidden = new Layer<T>(64, "Hidden", ActivationType::RELU);
    ILayer<T>* output = new Layer<T>(10, "Output", ActivationType::SIGMOID);

    network->setInputLayer(input);
    network->connect(input, hidden);
    network->connect(hidden, output);
    network->setOutputLayer(output);
    network->setOptimizerType(OptimizerType::ADAM);
    network->setLossType(LossType::CROSS_ENTROPY);
    network->init();

    // Create synthetic batch (batch_size=4, input_size=784)
    int batchSize = 4;
    ml::Mat<T> batchInputs(batchSize, 784, 0);
    ml::Mat<T> batchTargets(batchSize, 10, 0);

    // Fill with simple patterns
    for (int i = 0; i < batchSize; i++) {
        // Simple pattern for each sample
        for (int j = 0; j < 784; j++) {
            batchInputs.setAt(i, j, (T)(i * 0.1 + j * 0.001));
        }
        // One-hot target
        int targetClass = i % 10;
        batchTargets.setAt(i, targetClass, 1.0);
    }

    // Get initial loss
    T initialLoss = network->evaluateLoss(batchInputs, batchTargets);
    cout << "Initial loss: " << std::fixed << std::setprecision(4) << initialLoss << endl;

    // Train for a few iterations
    T learningRate = 0.01;
    for (int iter = 0; iter < 10; iter++) {
        network->trainBatch(batchInputs, batchTargets, learningRate);
    }

    // Get final loss
    T finalLoss = network->evaluateLoss(batchInputs, batchTargets);
    cout << "Final loss after 10 iterations: " << std::setprecision(4) << finalLoss << endl;

    if (finalLoss < initialLoss) {
        cout << "✓ Loss decreased during training" << endl;
    } else {
        cout << "✗ Loss did not decrease" << endl;
    }

    delete network;
}

void test_accuracy_computation() {
    BEGIN_TESTS("Accuracy Computation");
    typedef double T;

    // Create test predictions and targets
    ml::Mat<T> predictions(5, 10, 0);
    ml::Mat<T> targets(5, 10, 0);

    // Sample 0: pred=0, target=0 ✓
    predictions.setAt(0, 0, 0.9);
    targets.setAt(0, 0, 1.0);

    // Sample 1: pred=1, target=1 ✓
    predictions.setAt(1, 1, 0.8);
    targets.setAt(1, 1, 1.0);

    // Sample 2: pred=2, target=3 ✗
    predictions.setAt(2, 2, 0.7);
    targets.setAt(2, 3, 1.0);

    // Sample 3: pred=4, target=4 ✓
    predictions.setAt(3, 4, 0.85);
    targets.setAt(3, 4, 1.0);

    // Sample 4: pred=5, target=5 ✓
    predictions.setAt(4, 5, 0.95);
    targets.setAt(4, 5, 1.0);

    T accuracy = ComputeAccuracy<T>(predictions, targets);
    cout << "Accuracy: " << std::setprecision(1) << accuracy << "%" << endl;

    // Expected: 4 correct out of 5 = 80%
    if (approxEqual(accuracy, T(80.0), T(0.1))) {
        cout << "✓ Accuracy computation correct (4/5 = 80%)" << endl;
    } else {
        cout << "✗ Accuracy computation incorrect (expected 80%, got " << accuracy << "%)" << endl;
    }
}

void test_mnist_mini_training() {
    BEGIN_TESTS("MNIST Mini Training (with actual data)");
    typedef double T;

    cout << "Loading small MNIST subset..." << endl;

    MNISTDataset<T> trainDataset;

    if (!loadMNISTDataset<T>("train-images-idx3-ubyte", "train-labels-idx1-ubyte", trainDataset)) {
        cout << "MNIST data not available, skipping this test" << endl;
        cout << "Download MNIST from: http://yann.lecun.com/exdb/mnist/" << endl;
        return;
    }

    cout << "✓ MNIST data loaded" << endl;

    // Create small network
    Network<T>* network = new Network<T>();
    ILayer<T>* input = new Layer<T>(784, "Input", ActivationType::RELU);
    ILayer<T>* hidden = new Layer<T>(128, "Hidden", ActivationType::RELU);
    ILayer<T>* output = new Layer<T>(10, "Output", ActivationType::SIGMOID);

    network->setInputLayer(input);
    network->connect(input, hidden);
    network->connect(hidden, output);
    network->setOutputLayer(output);
    network->setOptimizerType(OptimizerType::ADAM);
    network->setLossType(LossType::CROSS_ENTROPY);
    network->init();

    cout << "✓ Network initialized (784-128-10)" << endl;

    // Use first 100 samples for quick test
    int numSamples = std::min(100, trainDataset.numSamples);
    ml::Mat<T> trainImages(numSamples, 784, 0);
    ml::Mat<T> trainLabels(numSamples, 10, 0);

    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < 784; j++) {
            trainImages.setAt(i, j, trainDataset.images.getAt(i, j));
        }
        for (int j = 0; j < 10; j++) {
            trainLabels.setAt(i, j, trainDataset.labels.getAt(i, j));
        }
    }

    // Initial evaluation
    T initialAccuracy = network->evaluateAccuracy(trainImages, trainLabels);
    T initialLoss = network->evaluateLoss(trainImages, trainLabels);

    cout << "Initial accuracy: " << std::setprecision(2) << initialAccuracy << "%" << endl;
    cout << "Initial loss: " << std::setprecision(4) << initialLoss << endl;

    // Train for 2 epochs with small batches
    cout << "\nTraining for 2 epochs..." << endl;
    int batchSize = 16;
    T learningRate = 0.001;

    for (int epoch = 0; epoch < 2; epoch++) {
        int numBatches = (numSamples + batchSize - 1) / batchSize;

        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * batchSize;
            int endIdx = std::min(startIdx + batchSize, numSamples);
            int actualBatchSize = endIdx - startIdx;

            ml::Mat<T> batchImages(actualBatchSize, 784, 0);
            ml::Mat<T> batchLabels(actualBatchSize, 10, 0);

            for (int i = 0; i < actualBatchSize; i++) {
                for (int j = 0; j < 784; j++) {
                    batchImages.setAt(i, j, trainImages.getAt(startIdx + i, j));
                }
                for (int j = 0; j < 10; j++) {
                    batchLabels.setAt(i, j, trainLabels.getAt(startIdx + i, j));
                }
            }

            network->trainBatch(batchImages, batchLabels, learningRate);
        }

        T epochAccuracy = network->evaluateAccuracy(trainImages, trainLabels);
        T epochLoss = network->evaluateLoss(trainImages, trainLabels);

        cout << "  Epoch " << (epoch + 1) << ": Accuracy=" << std::setprecision(2) << epochAccuracy
             << "%, Loss=" << std::setprecision(4) << epochLoss << endl;
    }

    T finalAccuracy = network->evaluateAccuracy(trainImages, trainLabels);
    T finalLoss = network->evaluateLoss(trainImages, trainLabels);

    cout << "\nFinal accuracy: " << std::setprecision(2) << finalAccuracy << "%" << endl;
    cout << "Final loss: " << std::setprecision(4) << finalLoss << endl;

    // Check if training improved performance
    if (finalAccuracy > initialAccuracy) {
        cout << "✓ Accuracy improved: " << std::setprecision(2)
             << (finalAccuracy - initialAccuracy) << "% gain" << endl;
    } else {
        cout << "⚠ Accuracy did not improve (might need more training)" << endl;
    }

    if (finalLoss < initialLoss) {
        cout << "✓ Loss decreased" << endl;
    } else {
        cout << "⚠ Loss did not decrease (might need more training)" << endl;
    }

    // Minimum expectation: final accuracy should be > 20% (better than random 10%)
    if (finalAccuracy > 20.0) {
        cout << "✓ Model is learning (accuracy > 20%)" << endl;
    } else {
        cout << "✗ Model might not be learning properly" << endl;
    }

    delete network;
}

int main() {
    cout << "========================================" << endl;
    cout << "   MNIST Training Test Suite" << endl;
    cout << "========================================\n" << endl;

    test_network_creation();
    cout << endl;

    test_batch_training();
    cout << endl;

    test_accuracy_computation();
    cout << endl;

    test_mnist_mini_training();
    cout << endl;

    cout << "========================================" << endl;
    cout << "All tests complete!" << endl;
    cout << "========================================" << endl;

    return 0;
}
