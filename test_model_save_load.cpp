#include <iostream>
#include <cmath>
#include "allheader.h"
#include "network.h"

using namespace ml;

// Helper function to compare matrices with tolerance
template <typename T>
bool matricesEqual(const Mat<T>& m1, const Mat<T>& m2, T tolerance = 1e-6) {
    if (m1.size().cx != m2.size().cx || m1.size().cy != m2.size().cy) {
        return false;
    }

    for (int i = 0; i < m1.size().cy; ++i) {
        for (int j = 0; j < m1.size().cx; ++j) {
            if (std::abs(m1.getAt(i, j) - m2.getAt(i, j)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    std::cout << "=== Neural Network Save/Load Test ===" << std::endl;

    // Create a simple 3-layer network: 2 inputs -> 3 hidden -> 1 output
    Network<double>* network1 = new Network<double>();

    ILayer<double>* inputLayer1  = new Layer<double>(2, "Input");
    ILayer<double>* hiddenLayer1 = new Layer<double>(3, "Hidden");
    ILayer<double>* outputLayer1 = new Layer<double>(1, "Output");

    // Connect layers
    network1->setInputLayer(inputLayer1);
    network1->connect(inputLayer1, hiddenLayer1);
    network1->connect(hiddenLayer1, outputLayer1);
    network1->setOutputLayer(outputLayer1);

    // Initialize weights
    std::cout << "\nInitializing network 1..." << std::endl;
    network1->init();

    // Test forward pass with original network
    std::cout << "\nTesting forward pass with original network..." << std::endl;
    Mat<double> testInput(1, 2, 0);
    testInput.setAt(0, 0, 0.5);
    testInput.setAt(0, 1, 0.8);

    Mat<double> output1 = network1->feed(testInput);
    std::cout << "Original network output: " << output1.getAt(0, 0) << std::endl;

    // Save the model
    std::string filename = "test_model.json";
    std::cout << "\nSaving model to " << filename << "..." << std::endl;
    if (!network1->saveToFile(filename)) {
        std::cerr << "Failed to save model!" << std::endl;
        delete network1;
        return 1;
    }

    // Create a second network with identical structure
    std::cout << "\nCreating second network with same structure..." << std::endl;
    Network<double>* network2 = new Network<double>();

    ILayer<double>* inputLayer2  = new Layer<double>(2, "Input");
    ILayer<double>* hiddenLayer2 = new Layer<double>(3, "Hidden");
    ILayer<double>* outputLayer2 = new Layer<double>(1, "Output");

    network2->setInputLayer(inputLayer2);
    network2->connect(inputLayer2, hiddenLayer2);
    network2->connect(hiddenLayer2, outputLayer2);
    network2->setOutputLayer(outputLayer2);

    // Initialize with random weights (different from network1)
    std::cout << "Initializing network 2 with random weights..." << std::endl;
    network2->init();

    // Test forward pass before loading
    Mat<double> output2_before = network2->feed(testInput);
    std::cout << "Network 2 output (before loading): " << output2_before.getAt(0, 0) << std::endl;

    // Load weights from file
    std::cout << "\nLoading model from " << filename << "..." << std::endl;
    if (!network2->loadFromFile(filename)) {
        std::cerr << "Failed to load model!" << std::endl;
        delete network1;
        delete network2;
        return 1;
    }

    // Test forward pass after loading
    std::cout << "\nTesting forward pass with loaded network..." << std::endl;
    Mat<double> output2_after = network2->feed(testInput);
    std::cout << "Network 2 output (after loading): " << output2_after.getAt(0, 0) << std::endl;

    // Verify outputs match
    std::cout << "\nVerifying outputs match..." << std::endl;
    if (matricesEqual(output1, output2_after, 1e-9)) {
        std::cout << "SUCCESS: Outputs match! Save/load working correctly." << std::endl;
    } else {
        std::cout << "FAILURE: Outputs don't match!" << std::endl;
        std::cout << "  Original: " << output1.getAt(0, 0) << std::endl;
        std::cout << "  Loaded:   " << output2_after.getAt(0, 0) << std::endl;
        std::cout << "  Diff:     " << std::abs(output1.getAt(0, 0) - output2_after.getAt(0, 0)) << std::endl;
        delete network1;
        delete network2;
        return 1;
    }

    // Test with multiple inputs
    std::cout << "\nTesting with multiple inputs..." << std::endl;
    Mat<double> inputs[] = {
        Mat<double>(1, 2, 0),
        Mat<double>(1, 2, 0),
        Mat<double>(1, 2, 0)
    };

    inputs[0].setAt(0, 0, 0.0); inputs[0].setAt(0, 1, 0.0);
    inputs[1].setAt(0, 0, 1.0); inputs[1].setAt(0, 1, 0.0);
    inputs[2].setAt(0, 0, 0.0); inputs[2].setAt(0, 1, 1.0);

    bool allMatch = true;
    for (int i = 0; i < 3; ++i) {
        Mat<double> out1 = network1->feed(inputs[i]);
        Mat<double> out2 = network2->feed(inputs[i]);

        if (!matricesEqual(out1, out2, 1e-9)) {
            std::cout << "  Input " << i << " - MISMATCH" << std::endl;
            allMatch = false;
        } else {
            std::cout << "  Input " << i << " - Match (output: " << out1.getAt(0, 0) << ")" << std::endl;
        }
    }

    if (allMatch) {
        std::cout << "\nAll tests PASSED!" << std::endl;
    } else {
        std::cout << "\nSome tests FAILED!" << std::endl;
    }

    // Cleanup
    delete network1;
    delete network2;

    return allMatch ? 0 : 1;
}
