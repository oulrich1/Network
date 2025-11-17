#include <iostream>
#include <cmath>
#include <cassert>
#include "allheader.h"
#include "network.h"

using namespace ml;
using namespace std;

int main() {
    cout << "=== HYPOTHESIS: Output error needs sigmoid derivative ===" << endl;

    typedef double T;

    // Create simple XOR network
    Network<T>* network = new Network<T>();
    ILayer<T>* inputLayer = new Layer<T>(2, "Input");
    ILayer<T>* hiddenLayer = new Layer<T>(4, "Hidden");
    ILayer<T>* outputLayer = new Layer<T>(1, "Output");

    network->setInputLayer(inputLayer);
    network->connect(inputLayer, hiddenLayer);
    network->connect(hiddenLayer, outputLayer);
    network->setOutputLayer(outputLayer);
    network->init();

    // XOR training data
    vector<Mat<T>> inputs;
    vector<Mat<T>> expected;

    Mat<T> in1(1, 2, 0); in1.setAt(0, 0, 0.0); in1.setAt(0, 1, 0.0);
    inputs.push_back(in1); expected.push_back(Mat<T>(1, 1, 0.0));

    Mat<T> in2(1, 2, 0); in2.setAt(0, 0, 0.0); in2.setAt(0, 1, 1.0);
    inputs.push_back(in2); expected.push_back(Mat<T>(1, 1, 1.0));

    Mat<T> in3(1, 2, 0); in3.setAt(0, 0, 1.0); in3.setAt(0, 1, 0.0);
    inputs.push_back(in3); expected.push_back(Mat<T>(1, 1, 1.0));

    Mat<T> in4(1, 2, 0); in4.setAt(0, 0, 1.0); in4.setAt(0, 1, 1.0);
    inputs.push_back(in4); expected.push_back(Mat<T>(1, 1, 0.0));

    const int epochs = 5000;
    const T learningRate = 1.0;  // Higher learning rate

    cout << "Training with FIXED error (including sigmoid derivative)..." << endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        T totalError = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            Mat<T> output = network->feed(inputs[i]);

            // Compute error WITH sigmoid derivative
            Mat<T> rawError = Diff<T>(expected[i], output);  // (target - output)

            // Apply sigmoid derivative: error * output * (1 - output)
            Mat<T> errorWithDerivative(rawError.size(), 0);
            for (int r = 0; r < rawError.size().cy; ++r) {
                for (int c = 0; c < rawError.size().cx; ++c) {
                    T err = rawError.getAt(r, c);
                    T out = output.getAt(r, c);
                    T derivative = out * (1.0 - out);  // sigmoid derivative
                    errorWithDerivative.setAt(r, c, err * derivative);
                }
            }

            T sampleError = 0;
            for (int j = 0; j < rawError.size().cx; ++j) {
                for (int k = 0; k < rawError.size().cy; ++k) {
                    T err = rawError.getAt(k, j);
                    sampleError += err * err;
                }
            }
            totalError += sampleError;

            // Use error WITH derivative for backprop
            outputLayer->setErrors(errorWithDerivative);
            network->backprop();
            network->updateWeights(learningRate);
        }

        if (epoch % 1000 == 0 || epoch == epochs - 1) {
            cout << "Epoch " << epoch << " - Error: " << totalError << endl;
        }
    }

    cout << "\nTesting network:" << endl;
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Mat<T> output = network->feed(inputs[i]);
        T predicted = output.getAt(0, 0);
        T target = expected[i].getAt(0, 0);
        T predictedClass = (predicted > 0.5) ? 1.0 : 0.0;

        const char* labels[] = {"[0,0]->0", "[0,1]->1", "[1,0]->1", "[1,1]->0"};
        cout << labels[i] << " : output=" << predicted
             << " predicted=" << predictedClass
             << " (target=" << target << ")" << endl;

        if (abs(predictedClass - target) < 0.1) {
            correct++;
        }
    }

    T accuracy = (100.0 * correct) / inputs.size();
    cout << "\nAccuracy: " << accuracy << "% (" << correct << "/4)" << endl;

    delete network;
    delete outputLayer;
    delete hiddenLayer;
    delete inputLayer;

    if (accuracy > 90.0) {
        cout << "\nSUCCESS: Adding sigmoid derivative FIXED the learning!" << endl;
        return 0;
    } else {
        cout << "\nFAILED: Sigmoid derivative didn't help." << endl;
        return 1;
    }
}
