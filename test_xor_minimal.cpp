#include <iostream>
#include "allheader.h"
#include "network.h"

using namespace std;
using namespace ml;

int main() {
    typedef double T;

    // Create XOR network
    Network<T>* net = new Network<T>();
    ILayer<T>* input = new Layer<T>(2, "I");
    ILayer<T>* hidden = new Layer<T>(3, "H");
    ILayer<T>* output = new Layer<T>(1, "O");

    net->setInputLayer(input);
    net->connect(input, hidden);
    net->connect(hidden, output);
    net->setOutputLayer(output);
    net->init();

    // XOR data
    Mat<T> x1(1, 2, 0); x1.setAt(0, 0, 0); x1.setAt(0, 1, 0);
    Mat<T> x2(1, 2, 0); x2.setAt(0, 0, 0); x2.setAt(0, 1, 1);
    Mat<T> x3(1, 2, 0); x3.setAt(0, 0, 1); x3.setAt(0, 1, 0);
    Mat<T> x4(1, 2, 0); x4.setAt(0, 0, 1); x4.setAt(0, 1, 1);

    Mat<T> y1(1, 1, 0);
    Mat<T> y2(1, 1, 1);
    Mat<T> y3(1, 1, 1);
    Mat<T> y4(1, 1, 0);

    cout << "Initial predictions:" << endl;
    cout << "[0,0] -> " << net->feed(x1).getAt(0, 0) << " (expect 0)" << endl;
    cout << "[0,1] -> " << net->feed(x2).getAt(0, 0) << " (expect 1)" << endl;
    cout << "[1,0] -> " << net->feed(x3).getAt(0, 0) << " (expect 1)" << endl;
    cout << "[1,1] -> " << net->feed(x4).getAt(0, 0) << " (expect 0)" << endl;

    T lr = 0.5;
    cout << "\nTraining..." << endl;

    for (int epoch = 0; epoch < 1000; ++epoch) {
        T totalErr = 0;

        // Sample 1
        Mat<T> p1 = net->feed(x1);
        Mat<T> e1 = Diff(y1, p1);
        totalErr += e1.getAt(0, 0) * e1.getAt(0, 0);
        output->setErrors(e1);
        net->backprop();
        net->updateWeights(lr);

        // Sample 2
        Mat<T> p2 = net->feed(x2);
        Mat<T> e2 = Diff(y2, p2);
        totalErr += e2.getAt(0, 0) * e2.getAt(0, 0);
        output->setErrors(e2);
        net->backprop();
        net->updateWeights(lr);

        // Sample 3
        Mat<T> p3 = net->feed(x3);
        Mat<T> e3 = Diff(y3, p3);
        totalErr += e3.getAt(0, 0) * e3.getAt(0, 0);
        output->setErrors(e3);
        net->backprop();
        net->updateWeights(lr);

        // Sample 4
        Mat<T> p4 = net->feed(x4);
        Mat<T> e4 = Diff(y4, p4);
        totalErr += e4.getAt(0, 0) * e4.getAt(0, 0);
        output->setErrors(e4);
        net->backprop();
        net->updateWeights(lr);

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << ": Error = " << totalErr;
            cout << ", Predictions: " << p1.getAt(0, 0) << " " << p2.getAt(0, 0)
                 << " " << p3.getAt(0, 0) << " " << p4.getAt(0, 0) << endl;
        }
    }

    cout << "\nFinal predictions:" << endl;
    T pred1 = net->feed(x1).getAt(0, 0);
    T pred2 = net->feed(x2).getAt(0, 0);
    T pred3 = net->feed(x3).getAt(0, 0);
    T pred4 = net->feed(x4).getAt(0, 0);

    cout << "[0,0] -> " << pred1 << " (expect 0)" << endl;
    cout << "[0,1] -> " << pred2 << " (expect 1)" << endl;
    cout << "[1,0] -> " << pred3 << " (expect 1)" << endl;
    cout << "[1,1] -> " << pred4 << " (expect 0)" << endl;

    int correct = 0;
    if (pred1 < 0.5) correct++;
    if (pred2 > 0.5) correct++;
    if (pred3 > 0.5) correct++;
    if (pred4 < 0.5) correct++;

    cout << "\nAccuracy: " << (100.0 * correct / 4) << "%" << endl;

    delete net;
    return 0;
}
