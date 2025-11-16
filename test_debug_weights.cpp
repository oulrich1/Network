#include <iostream>
#include "allheader.h"
#include "network.h"

using namespace std;
using namespace ml;

void printWeights(const char* label, Mat<double> w) {
    cout << label << " (" << w.size().cy << "x" << w.size().cx << "):" << endl;
    for (int i = 0; i < min(w.size().cy, 3); ++i) {
        cout << "  ";
        for (int j = 0; j < min(w.size().cx, 5); ++j) {
            cout << w.getAt(i, j) << " ";
        }
        cout << endl;
    }
}

int main() {
    typedef double T;

    cout << "Creating 2-2-1 network..." << endl;
    Network<T>* net = new Network<T>();
    ILayer<T>* input = new Layer<T>(2, "I");
    ILayer<T>* hidden = new Layer<T>(2, "H");
    ILayer<T>* output = new Layer<T>(1, "O");

    net->setInputLayer(input);
    net->connect(input, hidden);
    net->connect(hidden, output);
    net->setOutputLayer(output);
    net->init();

    // Single training example: [1, 0] -> 1
    Mat<T> x(1, 2, 0);
    x.setAt(0, 0, 1.0);
    x.setAt(0, 1, 0.0);
    Mat<T> y(1, 1, 1.0);

    T lr = 0.1;

    printWeights("\nInitial I->H weights", input->getWeights(hidden));
    printWeights("Initial H->O weights", hidden->getWeights(output));

    for (int iter = 0; iter < 5; ++iter) {
        cout << "\n=== Iteration " << iter << " ===" << endl;

        // Forward
        Mat<T> pred = net->feed(x);
        cout << "Prediction: " << pred.getAt(0, 0) << endl;

        // Compute error
        Mat<T> err = Diff(y, pred);
        cout << "Error: " << err.getAt(0, 0) << " (target - pred = " << y.getAt(0, 0) << " - " << pred.getAt(0, 0) << ")" << endl;

        // Get activations before backprop
        Mat<T> hiddenAct = hidden->getActivatedInput();
        cout << "Hidden activations: ";
        for (int i = 0; i < hiddenAct.size().cx; ++i) {
            cout << hiddenAct.getAt(0, i) << " ";
        }
        cout << endl;

        // Backprop
        output->setErrors(err);
        net->backprop();

        // Check propagated errors
        Mat<T> hiddenErr = hidden->getErrors();
        cout << "Hidden errors after backprop: ";
        for (int i = 0; i < hiddenErr.size().cx; ++i) {
            cout << hiddenErr.getAt(0, i) << " ";
        }
        cout << endl;

        // Get weights before update
        Mat<T> oldWeightsHO = hidden->getWeights(output);

        // MANUAL weight update check
        cout << "Manual check - what SHOULD happen:" << endl;
        Mat<T> errCopy = err.Copy();
        cout << "  Error dimensions: (" << errCopy.size().cy << "," << errCopy.size().cx << ")" << endl;
        cout << "  Hidden act dimensions: (" << hiddenAct.size().cy << "," << hiddenAct.size().cx << ")" << endl;
        cout << "  Expected delta[0,0] = " << err.getAt(0,0) << " * " << hiddenAct.getAt(0,0) << " = " << (err.getAt(0,0) * hiddenAct.getAt(0,0)) << endl;
        cout << "  Expected delta[0,1] = " << err.getAt(0,0) << " * " << hiddenAct.getAt(0,1) << " = " << (err.getAt(0,0) * hiddenAct.getAt(0,1)) << endl;
        cout << "  Expected delta[0,2] (bias) = " << err.getAt(0,0) << " * 1.0 = " << err.getAt(0,0) << endl;

        // Update
        net->updateWeights(lr);

        // Check weight changes
        Mat<T> newWeightsHO = hidden->getWeights(output);
        cout << "Actual H->O weight changes: ";
        for (int i = 0; i < min(newWeightsHO.size().cy, 2); ++i) {
            for (int j = 0; j < min(newWeightsHO.size().cx, 3); ++j) {
                T delta = newWeightsHO.getAt(i, j) - oldWeightsHO.getAt(i, j);
                cout << "(" << i << "," << j << "):" << delta << " ";
            }
        }
        cout << endl;
    }

    printWeights("\nFinal I->H weights", input->getWeights(hidden));
    printWeights("Final H->O weights", hidden->getWeights(output));

    delete net;
    return 0;
}
