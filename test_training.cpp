#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <cmath>
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
    const int epochs = 2000;
    const T learningRate = 0.5;

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
        test_xor_training();
        cout << endl;
        test_and_gate_training();
        cout << endl;
        test_or_gate_training();

        cout << endl;
        cout << "==================================================" << endl;
        cout << "    ALL TRAINING TESTS PASSED!" << endl;
        cout << "    All networks achieved >90% accuracy" << endl;
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
