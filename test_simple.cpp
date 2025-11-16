#include <iostream>
#include <cassert>
#include "allheader.h"
#include "network.h"

using namespace std;
using namespace ml;

int main() {
    typedef double T;

    cout << "Creating simple 2-2-1 network..." << endl;
    Network<T>* network = new Network<T>();
    ILayer<T>* input = new Layer<T>(2, "Input");
    ILayer<T>* hidden = new Layer<T>(2, "Hidden");
    ILayer<T>* output = new Layer<T>(1, "Output");

    network->setInputLayer(input);
    network->connect(input, hidden);
    network->connect(hidden, output);
    network->setOutputLayer(output);
    network->init();

    // Simple test: input [1, 0], expected output [1]
    Mat<T> testInput(1, 2, 0);
    testInput.setAt(0, 0, 1.0);
    testInput.setAt(0, 1, 0.0);

    Mat<T> expectedOutput(1, 1, 1.0);

    cout << "\nBefore training:" << endl;
    Mat<T> pred1 = network->feed(testInput);
    cout << "Input: [" << testInput.getAt(0, 0) << ", " << testInput.getAt(0, 1) << "]" << endl;
    cout << "Output: " << pred1.getAt(0, 0) << " (expected: 1.0)" << endl;

    // Get initial weights
    Mat<T> weights_ih = input->getWeights(hidden);
    cout << "\nInitial weights (input->hidden): " << weights_ih.size().cy << "x" << weights_ih.size().cx << endl;
    for (int i = 0; i < weights_ih.size().cy && i < 3; ++i) {
        cout << "  Row " << i << ": ";
        for (int j = 0; j < weights_ih.size().cx && j < 5; ++j) {
            cout << weights_ih.getAt(i, j) << " ";
        }
        cout << endl;
    }

    // Training loop
    const T learningRate = 0.1;
    cout << "\nTraining for 10 iterations..." << endl;
    for (int iter = 0; iter < 10; ++iter) {
        Mat<T> pred = network->feed(testInput);
        Mat<T> error = Diff<T>(expectedOutput, pred);

        T errorVal = error.getAt(0, 0);
        T predVal = pred.getAt(0, 0);

        cout << "Iter " << iter << ": pred=" << predVal << ", error=" << errorVal << endl;

        output->setErrors(error);
        network->backprop();

        // Check errors propagated to hidden layer
        Mat<T> hiddenErrors = hidden->getErrors();
        if (hiddenErrors.IsGood()) {
            cout << "  Hidden errors: ";
            for (int j = 0; j < hiddenErrors.size().cx && j < 4; ++j) {
                cout << hiddenErrors.getAt(0, j) << " ";
            }
            cout << endl;
        }

        network->updateWeights(learningRate);

        // Check if weights changed
        Mat<T> newWeights = input->getWeights(hidden);
        bool changed = false;
        for (int i = 0; i < weights_ih.size().cy && i < 2; ++i) {
            for (int j = 0; j < weights_ih.size().cx && j < 2; ++j) {
                if (abs(weights_ih.getAt(i, j) - newWeights.getAt(i, j)) > 1e-10) {
                    changed = true;
                    break;
                }
            }
        }
        cout << "  Weights changed: " << (changed ? "YES" : "NO") << endl;
        weights_ih = newWeights;
    }

    cout << "\nFinal weights (input->hidden): " << endl;
    for (int i = 0; i < weights_ih.size().cy && i < 3; ++i) {
        cout << "  Row " << i << ": ";
        for (int j = 0; j < weights_ih.size().cx && j < 5; ++j) {
            cout << weights_ih.getAt(i, j) << " ";
        }
        cout << endl;
    }

    cout << "\nAfter training:" << endl;
    Mat<T> pred2 = network->feed(testInput);
    cout << "Output: " << pred2.getAt(0, 0) << " (expected: 1.0)" << endl;

    delete network;
    return 0;
}
