#include <iostream>
#include <cmath>
#include "allheader.h"
#include "network.h"

using namespace ml;
using namespace std;

void printMatrix(const Mat<double>& mat, const string& name) {
    cout << name << " (" << mat.size().cy << "x" << mat.size().cx << "):" << endl;
    for (int i = 0; i < mat.size().cy; i++) {
        cout << "  [ ";
        for (int j = 0; j < mat.size().cx; j++) {
            cout << mat.getAt(i, j) << " ";
        }
        cout << "]" << endl;
    }
}

int main() {
    cout << "=== GRADIENT CHECK: Verify backprop is computing gradients ===" << endl;

    // Create simple 2-2-1 network
    Layer<double>* input = new Layer<double>(2, "input");
    Layer<double>* hidden = new Layer<double>(2, "hidden");
    Layer<double>* output = new Layer<double>(1, "output");

    Network<double>* net = new Network<double>();
    net->setInputLayer(input);
    net->setOutputLayer(output);
    net->connect(input, hidden);
    net->connect(hidden, output);
    net->init();

    cout << "\n--- Set specific weights for reproducibility ---" << endl;

    // Input->Hidden weights: (2, 3) - 2 hidden nodes, 3 inputs (2 + bias)
    Mat<double> w_ih(2, 3, 0);
    w_ih.setAt(0, 0, 0.5);
    w_ih.setAt(0, 1, 0.5);
    w_ih.setAt(0, 2, 0.1);
    w_ih.setAt(1, 0, -0.5);
    w_ih.setAt(1, 1, -0.5);
    w_ih.setAt(1, 2, 0.2);
    input->setWeights(hidden, w_ih);
    printMatrix(w_ih, "Initial input->hidden weights");

    // Hidden->Output weights: (1, 3) - 1 output node, 3 inputs (2 + bias)
    Mat<double> w_ho(1, 3, 0);
    w_ho.setAt(0, 0, 1.0);
    w_ho.setAt(0, 1, 1.0);
    w_ho.setAt(0, 2, 0.5);
    hidden->setWeights(output, w_ho);
    printMatrix(w_ho, "Initial hidden->output weights");

    // Test input: [1.0, 0.0]
    Mat<double> inputData(1, 2, 0);
    inputData.setAt(0, 0, 1.0);
    inputData.setAt(0, 1, 0.0);
    printMatrix(inputData, "\nInput");

    // Forward pass
    Mat<double> result = net->feed(inputData);
    printMatrix(result, "Output");

    // Target: [0.0]
    Mat<double> target(1, 1, 0.0);
    Mat<double> error = Diff<double>(target, result);
    printMatrix(error, "Error (target - output)");

    // Backward pass
    output->setErrors(error);
    net->backprop();

    cout << "\n--- After backprop, check if layers have errors set ---" << endl;
    Mat<double> output_errors = output->getErrors();
    Mat<double> hidden_errors = hidden->getErrors();
    Mat<double> input_errors = input->getErrors();

    printMatrix(output_errors, "Output layer errors");
    printMatrix(hidden_errors, "Hidden layer errors");
    printMatrix(input_errors, "Input layer errors");

    cout << "\n--- Update weights with small learning rate ---" << endl;
    Mat<double> w_ih_before = input->getWeights(hidden);
    Mat<double> w_ho_before = hidden->getWeights(output);

    net->updateWeights(0.1);

    Mat<double> w_ih_after = input->getWeights(hidden);
    Mat<double> w_ho_after = hidden->getWeights(output);

    cout << "\nInput->Hidden weight changes:" << endl;
    for (int i = 0; i < w_ih_before.size().cy; i++) {
        cout << "  Node " << i << ": ";
        for (int j = 0; j < w_ih_before.size().cx; j++) {
            double delta = w_ih_after.getAt(i, j) - w_ih_before.getAt(i, j);
            cout << delta << " ";
        }
        cout << endl;
    }

    cout << "\nHidden->Output weight changes:" << endl;
    for (int i = 0; i < w_ho_before.size().cy; i++) {
        cout << "  Node " << i << ": ";
        for (int j = 0; j < w_ho_before.size().cx; j++) {
            double delta = w_ho_after.getAt(i, j) - w_ho_before.getAt(i, j);
            cout << delta << " ";
        }
        cout << endl;
    }

    // Check if ANY weights changed
    bool weights_changed = false;
    for (int i = 0; i < w_ih_before.size().cy; i++) {
        for (int j = 0; j < w_ih_before.size().cx; j++) {
            if (abs(w_ih_after.getAt(i, j) - w_ih_before.getAt(i, j)) > 1e-10) {
                weights_changed = true;
            }
        }
    }
    for (int i = 0; i < w_ho_before.size().cy; i++) {
        for (int j = 0; j < w_ho_before.size().cx; j++) {
            if (abs(w_ho_after.getAt(i, j) - w_ho_before.getAt(i, j)) > 1e-10) {
                weights_changed = true;
            }
        }
    }

    cout << "\n========================================" << endl;
    if (weights_changed) {
        cout << "RESULT: Weights ARE changing - backprop working" << endl;
    } else {
        cout << "RESULT: Weights NOT changing - backprop BROKEN!" << endl;
    }
    cout << "========================================" << endl;

    delete net;
    delete output;
    delete hidden;
    delete input;

    return weights_changed ? 0 : 1;
}
