#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include "allheader.h"
#include "network.h"

using namespace std;
using namespace ml;
using namespace Utility;

// Helper function to check if two values are approximately equal
template <typename T>
bool approxEqual(T a, T b, T epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

// Helper to print network output
template <typename T>
void printOutput(const char* label, Mat<T> input, Mat<T> output, T expected) {
    cout << label << " [" << input.getAt(0, 0) << ", " << input.getAt(0, 1)
         << "] -> " << output.getAt(0, 0) << " (expected: " << expected << ")" << endl;
}

// Test XOR with training to >90% accuracy
void test_xor_training() {
    BEGIN_TESTS("Testing XOR Network Training (>90% accuracy)");
    typedef double T;

    // Create network: 2 inputs -> 4 hidden -> 1 output
    Network<T>* network = new Network<T>();
    ILayer<T>* inputLayer = new Layer<T>(2, "Input");
    ILayer<T>* hiddenLayer = new Layer<T>(4, "Hidden");
    ILayer<T>* outputLayer = new Layer<T>(1, "Output");

    network->setInputLayer(inputLayer);
    network->connect(inputLayer, hiddenLayer);
    network->connect(hiddenLayer, outputLayer);
    network->setOutputLayer(outputLayer);
    network->init();

    cout << ">> Network initialized with 2-4-1 architecture" << endl;

    // XOR training data: [input1, input2] -> [expected_output]
    // XOR truth table:
    // 0 XOR 0 = 0
    // 0 XOR 1 = 1
    // 1 XOR 0 = 1
    // 1 XOR 1 = 0

    vector<Mat<T>> inputs;
    vector<Mat<T>> expected;

    Mat<T> input1(1, 2, 0);
    input1.setAt(0, 0, 0.0);
    input1.setAt(0, 1, 0.0);
    inputs.push_back(input1);
    Mat<T> exp1(1, 1, 0.0);
    expected.push_back(exp1);

    Mat<T> input2(1, 2, 0);
    input2.setAt(0, 0, 0.0);
    input2.setAt(0, 1, 1.0);
    inputs.push_back(input2);
    Mat<T> exp2(1, 1, 1.0);
    expected.push_back(exp2);

    Mat<T> input3(1, 2, 0);
    input3.setAt(0, 0, 1.0);
    input3.setAt(0, 1, 0.0);
    inputs.push_back(input3);
    Mat<T> exp3(1, 1, 1.0);
    expected.push_back(exp3);

    Mat<T> input4(1, 2, 0);
    input4.setAt(0, 0, 1.0);
    input4.setAt(0, 1, 1.0);
    inputs.push_back(input4);
    Mat<T> exp4(1, 1, 0.0);
    expected.push_back(exp4);

    // Training parameters
    const int epochs = 10000;
    const T learningRate = 0.1;  // Reduced for stability

    cout << ">> Training for " << epochs << " epochs with learning rate " << learningRate << endl;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        T totalError = 0;

        // Train on each sample
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            Mat<T> output = network->feed(inputs[i]);

            // Compute error
            Mat<T> error = Diff<T>(expected[i], output);
            T sampleError = 0;
            for (int j = 0; j < error.size().cx; ++j) {
                for (int k = 0; k < error.size().cy; ++k) {
                    T err = error.getAt(k, j);
                    sampleError += err * err;
                }
            }
            totalError += sampleError;

            // Backward pass
            outputLayer->setErrors(error);
            network->backprop();

            // Update weights
            network->updateWeights(learningRate);
        }

        // Print progress every 200 epochs
        if (epoch % 200 == 0 || epoch == epochs - 1) {
            cout << "Epoch " << epoch << " - Total Error: " << totalError << endl;
        }
    }

    cout << "\n>> Training complete. Testing network..." << endl;

    // Test the trained network
    int correctPredictions = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Mat<T> output = network->feed(inputs[i]);
        T predicted = output.getAt(0, 0);
        T target = expected[i].getAt(0, 0);

        // Round to nearest integer for classification
        T predictedClass = (predicted > 0.5) ? 1.0 : 0.0;

        const char* labels[] = {"[0,0]->0", "[0,1]->1", "[1,0]->1", "[1,1]->0"};
        cout << "  " << labels[i] << " : output=" << predicted
             << " predicted=" << predictedClass
             << " (target=" << target << ")" << endl;

        if (approxEqual(predictedClass, target, 0.1)) {
            correctPredictions++;
        }
    }

    T accuracy = (100.0 * correctPredictions) / inputs.size();
    cout << "\n>> Accuracy: " << accuracy << "% (" << correctPredictions
         << "/" << inputs.size() << " correct)" << endl;

    // Assert >90% accuracy
    assert(accuracy > 90.0);
    cout << ">> XOR training test PASSED (accuracy > 90%)" << endl;

    delete network;
}

// Test AND gate with training
void test_and_gate_training() {
    BEGIN_TESTS("Testing AND Gate Training (>90% accuracy)");
    typedef double T;

    // Create network: 2 inputs -> 2 hidden -> 1 output
    Network<T>* network = new Network<T>();
    ILayer<T>* inputLayer = new Layer<T>(2, "Input");
    ILayer<T>* hiddenLayer = new Layer<T>(2, "Hidden");
    ILayer<T>* outputLayer = new Layer<T>(1, "Output");

    network->setInputLayer(inputLayer);
    network->connect(inputLayer, hiddenLayer);
    network->connect(hiddenLayer, outputLayer);
    network->setOutputLayer(outputLayer);
    network->init();

    cout << ">> Network initialized with 2-2-1 architecture" << endl;

    // AND truth table: both inputs must be 1 for output to be 1
    vector<Mat<T>> inputs;
    vector<Mat<T>> expected;

    Mat<T> input1(1, 2, 0);
    input1.setAt(0, 0, 0.0); input1.setAt(0, 1, 0.0);
    inputs.push_back(input1);
    expected.push_back(Mat<T>(1, 1, 0.0));

    Mat<T> input2(1, 2, 0);
    input2.setAt(0, 0, 0.0); input2.setAt(0, 1, 1.0);
    inputs.push_back(input2);
    expected.push_back(Mat<T>(1, 1, 0.0));

    Mat<T> input3(1, 2, 0);
    input3.setAt(0, 0, 1.0); input3.setAt(0, 1, 0.0);
    inputs.push_back(input3);
    expected.push_back(Mat<T>(1, 1, 0.0));

    Mat<T> input4(1, 2, 0);
    input4.setAt(0, 0, 1.0); input4.setAt(0, 1, 1.0);
    inputs.push_back(input4);
    expected.push_back(Mat<T>(1, 1, 1.0));

    // Training parameters
    const int epochs = 5000;
    const T learningRate = 0.5;

    cout << ">> Training for " << epochs << " epochs with learning rate " << learningRate << endl;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Mat<T> output = network->feed(inputs[i]);
            Mat<T> error = Diff<T>(expected[i], output);
            outputLayer->setErrors(error);
            network->backprop();
            network->updateWeights(learningRate);
        }
    }

    cout << ">> Training complete. Testing network..." << endl;

    // Test the trained network
    int correctPredictions = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Mat<T> output = network->feed(inputs[i]);
        T predicted = output.getAt(0, 0);
        T target = expected[i].getAt(0, 0);
        T predictedClass = (predicted > 0.5) ? 1.0 : 0.0;

        const char* labels[] = {"[0,0]->0", "[0,1]->0", "[1,0]->0", "[1,1]->1"};
        cout << "  " << labels[i] << " : output=" << predicted
             << " predicted=" << predictedClass << endl;

        if (approxEqual(predictedClass, target, 0.1)) {
            correctPredictions++;
        }
    }

    T accuracy = (100.0 * correctPredictions) / inputs.size();
    cout << ">> Accuracy: " << accuracy << "%" << endl;
    assert(accuracy > 90.0);
    cout << ">> AND gate test PASSED" << endl;

    delete network;
}

// Test simple linear regression: y = 2x + 1 (normalized to [0,1] for sigmoid)
void test_linear_regression() {
    BEGIN_TESTS("Testing Linear Regression: y = 2x + 1 (normalized) (MSE < 0.01, Accuracy > 95%)");
    typedef double T;

    // Create network: 1 input -> 8 hidden -> 1 output
    Network<T>* network = new Network<T>();
    ILayer<T>* inputLayer = new Layer<T>(1, "Input");
    ILayer<T>* hiddenLayer = new Layer<T>(8, "Hidden");
    ILayer<T>* outputLayer = new Layer<T>(1, "Output");

    network->setInputLayer(inputLayer);
    network->connect(inputLayer, hiddenLayer);
    network->connect(hiddenLayer, outputLayer);
    network->setOutputLayer(outputLayer);
    network->init();

    cout << ">> Network initialized with 1-8-1 architecture" << endl;

    // Generate training data for y = 2x + 1, normalized to [0,1]
    // Original: y = 2x + 1, with x in [0,1], y in [1,3]
    // Normalized: y_norm = (y - 1) / 2, so y_norm in [0,1]
    // Using 10 evenly spaced points in [0, 1]
    vector<Mat<T>> inputs;
    vector<Mat<T>> expected;
    vector<T> originalTargets;  // Store original values for display

    for (int i = 0; i <= 10; ++i) {
        T x = i / 10.0;  // 0.0, 0.1, 0.2, ..., 1.0
        T y_original = 2.0 * x + 1.0;  // y = 2x + 1 (range [1,3])
        T y_normalized = (y_original - 1.0) / 2.0;  // Normalize to [0,1]

        Mat<T> input(1, 1, 0);
        input.setAt(0, 0, x);
        inputs.push_back(input);

        Mat<T> target(1, 1, 0);
        target.setAt(0, 0, y_normalized);
        expected.push_back(target);

        originalTargets.push_back(y_original);
    }

    cout << ">> Generated " << inputs.size() << " training samples for y = 2x + 1" << endl;
    cout << ">> Outputs normalized to [0,1] for sigmoid: y_norm = (2x + 1 - 1) / 2 = x" << endl;
    cout << ">> Sample data: x=0.0 -> y=1.0 (norm=0.0), x=0.5 -> y=2.0 (norm=0.5), x=1.0 -> y=3.0 (norm=1.0)" << endl;

    // Training parameters
    const int epochs = 50000;
    const T learningRate = 0.1;

    cout << ">> Training for " << epochs << " epochs with learning rate " << learningRate << endl;

    // Start timing training
    auto train_start = std::chrono::high_resolution_clock::now();

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        T totalError = 0;

        // Train on each sample
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            Mat<T> output = network->feed(inputs[i]);

            // Compute error
            Mat<T> error = Diff<T>(expected[i], output);
            T sampleError = error.getAt(0, 0) * error.getAt(0, 0);
            totalError += sampleError;

            // Backward pass
            outputLayer->setErrors(error);
            network->backprop();

            // Update weights
            network->updateWeights(learningRate);
        }

        // Print progress every 5000 epochs
        if (epoch % 5000 == 0 || epoch == epochs - 1) {
            T mse = totalError / inputs.size();
            cout << "Epoch " << epoch << " - MSE: " << mse << endl;
        }
    }

    // End timing training
    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
    cout << "\n>> Training complete. Time: " << train_duration.count() << " ms" << endl;
    cout << ">> Testing network..." << endl;

    // Start timing testing
    auto test_start = std::chrono::high_resolution_clock::now();

    // Test the trained network
    T totalSquaredError = 0;
    int withinTolerance = 0;
    const T tolerance = 0.05;  // 5% relative error tolerance

    for (size_t i = 0; i < inputs.size(); ++i) {
        Mat<T> output = network->feed(inputs[i]);
        T predicted_normalized = output.getAt(0, 0);
        T target_normalized = expected[i].getAt(0, 0);

        // Denormalize for display
        T predicted_original = predicted_normalized * 2.0 + 1.0;
        T target_original = originalTargets[i];

        // Calculate error on normalized values (what network actually trains on)
        T error_normalized = target_normalized - predicted_normalized;
        T squaredError = error_normalized * error_normalized;
        totalSquaredError += squaredError;

        // Check if within 5% relative error (on original scale for interpretability)
        T error_original = target_original - predicted_original;
        T relativeError = std::abs(error_original / target_original);
        if (relativeError < tolerance) {
            withinTolerance++;
        }

        T x = inputs[i].getAt(0, 0);
        cout << "  x=" << x << " : predicted=" << predicted_original
             << " (norm=" << predicted_normalized << ")"
             << ", target=" << target_original
             << " (norm=" << target_normalized << ")"
             << ", error=" << error_original
             << ", relative_error=" << (relativeError * 100.0) << "%" << endl;
    }

    // End timing testing
    auto test_end = std::chrono::high_resolution_clock::now();
    auto test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);

    T mse = totalSquaredError / inputs.size();
    T accuracy = (100.0 * withinTolerance) / inputs.size();

    cout << "\n>> Testing complete. Time: " << test_duration.count() << " ms" << endl;
    cout << ">> Final MSE (on normalized values): " << mse << endl;
    cout << ">> Accuracy (within 5% tolerance): " << accuracy << "% ("
         << withinTolerance << "/" << inputs.size() << " samples)" << endl;

    // Assert MSE < 0.01 and accuracy > 95%
    assert(mse < 0.01);
    assert(accuracy > 95.0);

    cout << ">> Linear regression test PASSED (MSE < 0.01, Accuracy > 95%)" << endl;

    delete network;
}

// Test OR gate with training
void test_or_gate_training() {
    BEGIN_TESTS("Testing OR Gate Training (>90% accuracy)");
    typedef double T;

    // Create network: 2 inputs -> 2 hidden -> 1 output
    Network<T>* network = new Network<T>();
    ILayer<T>* inputLayer = new Layer<T>(2, "Input");
    ILayer<T>* hiddenLayer = new Layer<T>(2, "Hidden");
    ILayer<T>* outputLayer = new Layer<T>(1, "Output");

    network->setInputLayer(inputLayer);
    network->connect(inputLayer, hiddenLayer);
    network->connect(hiddenLayer, outputLayer);
    network->setOutputLayer(outputLayer);
    network->init();

    cout << ">> Network initialized with 2-2-1 architecture" << endl;

    // OR truth table: output is 1 if any input is 1
    vector<Mat<T>> inputs;
    vector<Mat<T>> expected;

    Mat<T> input1(1, 2, 0);
    input1.setAt(0, 0, 0.0); input1.setAt(0, 1, 0.0);
    inputs.push_back(input1);
    expected.push_back(Mat<T>(1, 1, 0.0));

    Mat<T> input2(1, 2, 0);
    input2.setAt(0, 0, 0.0); input2.setAt(0, 1, 1.0);
    inputs.push_back(input2);
    expected.push_back(Mat<T>(1, 1, 1.0));

    Mat<T> input3(1, 2, 0);
    input3.setAt(0, 0, 1.0); input3.setAt(0, 1, 0.0);
    inputs.push_back(input3);
    expected.push_back(Mat<T>(1, 1, 1.0));

    Mat<T> input4(1, 2, 0);
    input4.setAt(0, 0, 1.0); input4.setAt(0, 1, 1.0);
    inputs.push_back(input4);
    expected.push_back(Mat<T>(1, 1, 1.0));

    // Training parameters
    const int epochs = 5000;
    const T learningRate = 0.5;

    cout << ">> Training for " << epochs << " epochs" << endl;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Mat<T> output = network->feed(inputs[i]);
            Mat<T> error = Diff<T>(expected[i], output);
            outputLayer->setErrors(error);
            network->backprop();
            network->updateWeights(learningRate);
        }
    }

    cout << ">> Training complete. Testing network..." << endl;

    // Test the trained network
    int correctPredictions = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Mat<T> output = network->feed(inputs[i]);
        T predicted = output.getAt(0, 0);
        T target = expected[i].getAt(0, 0);
        T predictedClass = (predicted > 0.5) ? 1.0 : 0.0;

        const char* labels[] = {"[0,0]->0", "[0,1]->1", "[1,0]->1", "[1,1]->1"};
        cout << "  " << labels[i] << " : output=" << predicted
             << " predicted=" << predictedClass << endl;

        if (approxEqual(predictedClass, target, 0.1)) {
            correctPredictions++;
        }
    }

    T accuracy = (100.0 * correctPredictions) / inputs.size();
    cout << ">> Accuracy: " << accuracy << "%" << endl;
    assert(accuracy > 90.0);
    cout << ">> OR gate test PASSED" << endl;

    delete network;
}

int main() {
    cout << "==================================================" << endl;
    cout << "    Neural Network Training Tests" << endl;
    cout << "    Testing simple patterns with >90% accuracy" << endl;
    cout << "==================================================" << endl;

    try {
        test_linear_regression();
        cout << endl;
        test_xor_training();
        cout << endl;
        test_and_gate_training();
        cout << endl;
        test_or_gate_training();

        cout << endl;
        cout << "==================================================" << endl;
        cout << "    ALL TRAINING TESTS PASSED!" << endl;
        cout << "    All networks achieved target accuracy/MSE" << endl;
        cout << "==================================================" << endl;
        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Test failed with exception: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cerr << "Test failed with unknown exception" << endl;
        return 1;
    }
}
