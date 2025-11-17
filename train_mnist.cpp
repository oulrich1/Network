#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <algorithm>
#include "allheader.h"
#include "network.h"
#include "mnist_loader.h"

using namespace std;
using namespace ml;
using namespace Utility;

/**
 * MNIST Training Script
 *
 * Trains a fully-connected neural network on the MNIST digit classification dataset.
 * Architecture: 784 (input) → 256 (ReLU) → 128 (ReLU) → 10 (Sigmoid + Cross-Entropy)
 *
 * Features:
 * - Batch training for faster convergence
 * - Adam optimizer for adaptive learning
 * - Cross-entropy loss for classification
 * - Training/validation monitoring
 * - Model checkpointing
 * - Accuracy evaluation
 *
 * Expected performance: 85-92% test accuracy after 10-20 epochs
 */

// Training configuration
struct TrainConfig {
    int epochs = 10;
    int batchSize = 32;
    double learningRate = 0.001;
    int validationInterval = 1;  // Evaluate every N epochs
    int saveInterval = 5;         // Save model every N epochs
    bool shuffle = true;          // Shuffle training data each epoch
    string modelSavePath = "mnist_model.json";
};

// Helper function to shuffle indices
void shuffleIndices(std::vector<int>& indices) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
}

int main(int argc, char* argv[]) {
    cout << "========================================" << endl;
    cout << "   MNIST Digit Classification Training  " << endl;
    cout << "========================================\n" << endl;

    typedef double T;
    TrainConfig config;

    // Parse command line arguments (optional)
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batchSize = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learningRate = std::atof(argv[++i]);
        } else if (arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --epochs N        Number of training epochs (default: 10)" << endl;
            cout << "  --batch-size N    Batch size (default: 32)" << endl;
            cout << "  --lr RATE         Learning rate (default: 0.001)" << endl;
            cout << "  --help            Show this message" << endl;
            return 0;
        }
    }

    cout << "Configuration:" << endl;
    cout << "  Epochs: " << config.epochs << endl;
    cout << "  Batch size: " << config.batchSize << endl;
    cout << "  Learning rate: " << config.learningRate << endl;
    cout << endl;

    // ========================================
    // Load MNIST Dataset
    // ========================================
    cout << "Loading MNIST dataset..." << endl;

    MNISTDataset<T> trainDataset, testDataset;

    if (!loadMNISTDataset<T>("train-images-idx3-ubyte", "train-labels-idx1-ubyte", trainDataset)) {
        cerr << "Failed to load training data. Make sure MNIST files are in the current directory." << endl;
        cerr << "Download from: http://yann.lecun.com/exdb/mnist/" << endl;
        return 1;
    }

    if (!loadMNISTDataset<T>("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", testDataset)) {
        cerr << "Failed to load test data." << endl;
        return 1;
    }

    cout << "\nDataset loaded successfully!" << endl;
    cout << "  Training samples: " << trainDataset.numSamples << endl;
    cout << "  Test samples: " << testDataset.numSamples << endl;
    cout << endl;

    // ========================================
    // Create Neural Network
    // ========================================
    cout << "Creating neural network..." << endl;

    Network<T>* network = new Network<T>();

    // Architecture: 784 → 256 → 128 → 10
    ILayer<T>* inputLayer = new Layer<T>(784, "Input", ActivationType::RELU);
    ILayer<T>* hidden1 = new Layer<T>(256, "Hidden1", ActivationType::RELU);
    ILayer<T>* hidden2 = new Layer<T>(128, "Hidden2", ActivationType::RELU);
    ILayer<T>* outputLayer = new Layer<T>(10, "Output", ActivationType::SIGMOID);

    network->setInputLayer(inputLayer);
    network->connect(inputLayer, hidden1);
    network->connect(hidden1, hidden2);
    network->connect(hidden2, outputLayer);
    network->setOutputLayer(outputLayer);

    // Configure optimizer and loss
    network->setOptimizerType(OptimizerType::ADAM);
    network->setLossType(LossType::CROSS_ENTROPY);

    network->init();

    cout << "Network architecture:" << endl;
    cout << "  Input: 784 neurons (28x28 pixels)" << endl;
    cout << "  Hidden1: 256 neurons (ReLU)" << endl;
    cout << "  Hidden2: 128 neurons (ReLU)" << endl;
    cout << "  Output: 10 neurons (Sigmoid)" << endl;
    cout << "  Optimizer: Adam" << endl;
    cout << "  Loss: Cross-Entropy" << endl;
    cout << endl;

    // ========================================
    // Training Loop
    // ========================================
    cout << "Starting training...\n" << endl;

    int numBatches = (trainDataset.numSamples + config.batchSize - 1) / config.batchSize;

    // Track best accuracy for model saving
    T bestTestAccuracy = 0.0;

    auto trainingStartTime = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        auto epochStartTime = std::chrono::high_resolution_clock::now();

        // Shuffle training data
        std::vector<int> indices(trainDataset.numSamples);
        for (int i = 0; i < trainDataset.numSamples; i++) {
            indices[i] = i;
        }
        if (config.shuffle) {
            shuffleIndices(indices);
        }

        cout << "Epoch " << (epoch + 1) << "/" << config.epochs << endl;

        // Training batches
        T epochLoss = 0.0;
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * config.batchSize;
            int endIdx = std::min(startIdx + config.batchSize, trainDataset.numSamples);
            int actualBatchSize = endIdx - startIdx;

            // Extract batch data
            ml::Mat<T> batchImages(actualBatchSize, trainDataset.imageSize, 0);
            ml::Mat<T> batchLabels(actualBatchSize, trainDataset.numClasses, 0);

            for (int i = 0; i < actualBatchSize; i++) {
                int sampleIdx = indices[startIdx + i];
                for (int j = 0; j < trainDataset.imageSize; j++) {
                    batchImages.setAt(i, j, trainDataset.images.getAt(sampleIdx, j));
                }
                for (int j = 0; j < trainDataset.numClasses; j++) {
                    batchLabels.setAt(i, j, trainDataset.labels.getAt(sampleIdx, j));
                }
            }

            // Train on batch
            network->trainBatch(batchImages, batchLabels, config.learningRate);

            // Print progress every 100 batches
            if ((batch + 1) % 100 == 0 || batch == numBatches - 1) {
                // Compute loss on current batch
                T batchLoss = network->evaluateLoss(batchImages, batchLabels);
                epochLoss += batchLoss * actualBatchSize;

                cout << "  Batch " << (batch + 1) << "/" << numBatches
                     << " - Loss: " << std::fixed << std::setprecision(4) << batchLoss;

                // Show progress bar
                int barWidth = 30;
                float progress = (float)(batch + 1) / numBatches;
                cout << " [";
                int pos = barWidth * progress;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) cout << "=";
                    else if (i == pos) cout << ">";
                    else cout << " ";
                }
                cout << "] " << int(progress * 100.0) << "%\r";
                cout.flush();
            }
        }
        cout << endl;

        epochLoss /= trainDataset.numSamples;

        auto epochEndTime = std::chrono::high_resolution_clock::now();
        auto epochDuration = std::chrono::duration_cast<std::chrono::seconds>(
            epochEndTime - epochStartTime).count();

        // Evaluate on training and test sets
        if ((epoch + 1) % config.validationInterval == 0) {
            cout << "  Evaluating..." << endl;

            // Sample a subset for faster evaluation (use first 1000 samples)
            int trainEvalSize = std::min(1000, trainDataset.numSamples);
            ml::Mat<T> trainEvalImages(trainEvalSize, trainDataset.imageSize, 0);
            ml::Mat<T> trainEvalLabels(trainEvalSize, trainDataset.numClasses, 0);

            for (int i = 0; i < trainEvalSize; i++) {
                for (int j = 0; j < trainDataset.imageSize; j++) {
                    trainEvalImages.setAt(i, j, trainDataset.images.getAt(i, j));
                }
                for (int j = 0; j < trainDataset.numClasses; j++) {
                    trainEvalLabels.setAt(i, j, trainDataset.labels.getAt(i, j));
                }
            }

            T trainAccuracy = network->evaluateAccuracy(trainEvalImages, trainEvalLabels);
            T testAccuracy = network->evaluateAccuracy(testDataset.images, testDataset.labels);

            cout << "  Train Loss: " << std::fixed << std::setprecision(4) << epochLoss << endl;
            cout << "  Train Accuracy: " << std::setprecision(2) << trainAccuracy << "%" << endl;
            cout << "  Test Accuracy: " << std::setprecision(2) << testAccuracy << "%" << endl;
            cout << "  Time: " << epochDuration << "s" << endl;

            // Save best model
            if (testAccuracy > bestTestAccuracy) {
                bestTestAccuracy = testAccuracy;
                string bestModelPath = "mnist_model_best.json";
                cout << "  New best accuracy! Saving to " << bestModelPath << endl;
                network->saveToFile(bestModelPath);
            }
        }

        // Save checkpoint
        if ((epoch + 1) % config.saveInterval == 0) {
            string checkpointPath = "mnist_model_epoch" + std::to_string(epoch + 1) + ".json";
            cout << "  Saving checkpoint to " << checkpointPath << endl;
            network->saveToFile(checkpointPath);
        }

        cout << endl;
    }

    auto trainingEndTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(
        trainingEndTime - trainingStartTime).count();

    // ========================================
    // Final Evaluation
    // ========================================
    cout << "========================================" << endl;
    cout << "Training Complete!" << endl;
    cout << "========================================" << endl;
    cout << "Total training time: " << totalDuration << "s" << endl;
    cout << "Best test accuracy: " << std::setprecision(2) << bestTestAccuracy << "%" << endl;

    // Final save
    cout << "\nSaving final model to " << config.modelSavePath << endl;
    network->saveToFile(config.modelSavePath);

    // Show some predictions
    cout << "\nSample predictions:" << endl;
    for (int i = 0; i < 5; i++) {
        auto sample = getSample<T>(testDataset, i);
        ml::Mat<T> predicted = network->feed(sample.first);

        // Find predicted class
        int predictedClass = 0;
        T maxProb = predicted.getAt(0, 0);
        for (int j = 1; j < 10; j++) {
            if (predicted.getAt(0, j) > maxProb) {
                maxProb = predicted.getAt(0, j);
                predictedClass = j;
            }
        }

        int trueClass = testDataset.rawLabels[i];
        cout << "  Sample " << i << ": True=" << trueClass
             << ", Predicted=" << predictedClass
             << ", Confidence=" << std::setprecision(1) << (maxProb * 100) << "%"
             << (predictedClass == trueClass ? " ✓" : " ✗") << endl;
    }

    // Cleanup
    delete network;

    cout << "\nDone!" << endl;
    return 0;
}
