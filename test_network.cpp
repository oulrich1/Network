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

// Test 1: Test Sigmoid function
void test_sigmoid() {
    BEGIN_TESTS("Testing Sigmoid Function");
    typedef double T;

    // Create a simple 2x2 matrix
    Mat<T> input(2, 2, 0);
    input.setAt(0, 0, 0.0);   // sigmoid(0) should be 0.5
    input.setAt(0, 1, 1.0);   // sigmoid(1) should be ~0.731
    input.setAt(1, 0, -1.0);  // sigmoid(-1) should be ~0.269
    input.setAt(1, 1, 2.0);   // sigmoid(2) should be ~0.881

    Mat<T> output = Sigmoid<T>(input);

    // Verify sigmoid outputs
    assert(approxEqual(output.getAt(0, 0), 0.5, 1e-3));
    assert(approxEqual(output.getAt(0, 1), 0.731, 1e-3));
    assert(approxEqual(output.getAt(1, 0), 0.269, 1e-3));
    assert(approxEqual(output.getAt(1, 1), 0.881, 1e-3));

    cout << ">> Sigmoid function test PASSED" << endl;
}

// Test 2: Test SigmoidGrad (Sigmoid Gradient) function
void test_sigmoid_gradient() {
    BEGIN_TESTS("Testing Sigmoid Gradient Function");
    typedef double T;

    // Create a matrix with sigmoid outputs
    Mat<T> sigmoidOutput(2, 2, 0);
    sigmoidOutput.setAt(0, 0, 0.5);
    sigmoidOutput.setAt(0, 1, 0.731);
    sigmoidOutput.setAt(1, 0, 0.269);
    sigmoidOutput.setAt(1, 1, 0.881);

    Mat<T> gradient = SigmoidGrad<T>(sigmoidOutput);

    // Sigmoid gradient is: sig(x) * (1 - sig(x))
    // For sig(x) = 0.5: grad = 0.5 * 0.5 = 0.25
    assert(approxEqual(gradient.getAt(0, 0), 0.25, 1e-3));

    cout << ">> Sigmoid gradient test PASSED" << endl;
}

// Test 3: Test ElementMult function
void test_element_mult() {
    BEGIN_TESTS("Testing Element-wise Multiplication");
    typedef double T;

    Mat<T> m1(2, 2, 0);
    m1.setAt(0, 0, 2.0);
    m1.setAt(0, 1, 3.0);
    m1.setAt(1, 0, 4.0);
    m1.setAt(1, 1, 5.0);

    Mat<T> m2(2, 2, 0);
    m2.setAt(0, 0, 1.0);
    m2.setAt(0, 1, 2.0);
    m2.setAt(1, 0, 3.0);
    m2.setAt(1, 1, 4.0);

    Mat<T> result = ElementMult<T>(m1, m2);

    // Verify element-wise multiplication
    assert(approxEqual(result.getAt(0, 0), 2.0, 1e-5));  // 2 * 1 = 2
    assert(approxEqual(result.getAt(0, 1), 6.0, 1e-5));  // 3 * 2 = 6
    assert(approxEqual(result.getAt(1, 0), 12.0, 1e-5)); // 4 * 3 = 12
    assert(approxEqual(result.getAt(1, 1), 20.0, 1e-5)); // 5 * 4 = 20

    cout << ">> Element-wise multiplication test PASSED" << endl;
}

// Test 4: Test simple forward pass
void test_forward_pass() {
    BEGIN_TESTS("Testing Neural Network Forward Pass");
    typedef double T;

    Network<T>* network = new Network<T>();
    ILayer<T>* l1 = new Layer<T>(2, "Input");
    ILayer<T>* l2 = new Layer<T>(3, "Hidden");
    ILayer<T>* l3 = new Layer<T>(1, "Output");

    network->setInputLayer(l1);
    network->connect(l1, l2);
    network->connect(l2, l3);
    network->setOutputLayer(l3);
    network->init();

    // Create simple input
    Mat<T> input(1, 2, 1.0);
    Mat<T> output = network->feed(input);

    // Output should be valid and have correct dimensions
    assert(output.IsGood());
    assert(output.size().cx == 1);  // 1 output node
    assert(output.size().cy == 1);  // 1 sample

    cout << ">> Forward pass test PASSED" << endl;

    delete network;
}

// Test 5: Test backward propagation
void test_backward_pass() {
    BEGIN_TESTS("Testing Neural Network Backward Propagation");
    typedef double T;

    Network<T>* network = new Network<T>();
    ILayer<T>* l1 = new Layer<T>(2, "Input");
    ILayer<T>* l2 = new Layer<T>(3, "Hidden");
    ILayer<T>* l3 = new Layer<T>(1, "Output");

    network->setInputLayer(l1);
    network->connect(l1, l2);
    network->connect(l2, l3);
    network->setOutputLayer(l3);
    network->init();

    // Forward pass
    Mat<T> input(1, 2, 1.0);
    Mat<T> output = network->feed(input);

    // Set output error
    Mat<T> targetOutput(1, 1, 0.5);
    Mat<T> error = Diff<T>(targetOutput, output);
    l3->setErrors(error);

    // Backward pass
    network->backprop();

    // Verify that errors have been propagated to hidden and input layers
    Mat<T> l2_errors = l2->getErrors();
    Mat<T> l1_errors = l1->getErrors();

    assert(l2_errors.IsGood());
    assert(l1_errors.IsGood());
    assert(l2_errors.size().cx == 3);  // hidden layer has 3 nodes
    assert(l1_errors.size().cx == 2);  // input layer has 2 nodes

    cout << ">> Backward propagation test PASSED" << endl;

    delete network;
}

// Test 6: Test layer connections and dependencies
void test_layer_dependencies() {
    BEGIN_TESTS("Testing Layer Dependencies");
    typedef double T;

    Network<T>* network = new Network<T>();
    ILayer<T>* l1 = new Layer<T>(5, "L1");
    ILayer<T>* l2 = new Layer<T>(3, "L2");
    ILayer<T>* l3 = new Layer<T>(2, "L3");

    network->setInputLayer(l1);
    network->connect(l1, l2);
    network->connect(l2, l3);
    network->setOutputLayer(l3);

    // Verify dependencies
    assert(l2->dependancies.size() == 1);
    assert(l2->dependancies[0] == l1);
    assert(l3->dependancies.size() == 1);
    assert(l3->dependancies[0] == l2);

    cout << ">> Layer dependencies test PASSED" << endl;

    delete network;
}

// Test 7: Test XOR problem (simple integration test)
void test_xor_network() {
    BEGIN_TESTS("Testing XOR Network Training");
    typedef double T;

    Network<T>* network = new Network<T>();
    ILayer<T>* l1 = new Layer<T>(2, "Input");
    ILayer<T>* l2 = new Layer<T>(4, "Hidden");  // Hidden layer with 4 nodes
    ILayer<T>* l3 = new Layer<T>(1, "Output");

    network->setInputLayer(l1);
    network->connect(l1, l2);
    network->connect(l2, l3);
    network->setOutputLayer(l3);
    network->init();

    // XOR training data
    Mat<T> input1(1, 2, 0);
    input1.setAt(0, 0, 0.0);
    input1.setAt(0, 1, 0.0);

    Mat<T> input2(1, 2, 0);
    input2.setAt(0, 0, 0.0);
    input2.setAt(0, 1, 1.0);

    Mat<T> input3(1, 2, 0);
    input3.setAt(0, 0, 1.0);
    input3.setAt(0, 1, 0.0);

    Mat<T> input4(1, 2, 0);
    input4.setAt(0, 0, 1.0);
    input4.setAt(0, 1, 1.0);

    // Test forward pass for all XOR inputs
    Mat<T> output1 = network->feed(input1);
    Mat<T> output2 = network->feed(input2);
    Mat<T> output3 = network->feed(input3);
    Mat<T> output4 = network->feed(input4);

    // All outputs should be valid
    assert(output1.IsGood());
    assert(output2.IsGood());
    assert(output3.IsGood());
    assert(output4.IsGood());

    cout << ">> XOR network test PASSED" << endl;

    delete network;
}

// Test 8: Test matrix operations used in backprop
void test_matrix_operations() {
    BEGIN_TESTS("Testing Matrix Operations for Backprop");
    typedef double T;

    // Test matrix transpose and multiply
    Mat<T> weights(2, 3, 0);
    weights.setAt(0, 0, 0.5);
    weights.setAt(0, 1, 0.3);
    weights.setAt(0, 2, 0.2);
    weights.setAt(1, 0, 0.4);
    weights.setAt(1, 1, 0.6);
    weights.setAt(1, 2, 0.1);

    Mat<T> errors(1, 2, 0);
    errors.setAt(0, 0, 0.1);
    errors.setAt(0, 1, 0.2);

    weights.Transpose();
    // Use standalone Mult function with bIsTransposedAlready=true
    Mat<T> weightedErr = ml::Mult<T>(weights, errors, true);

    assert(weightedErr.IsGood());
    assert(weightedErr.size().cy == 1);
    assert(weightedErr.size().cx == 3);

    cout << ">> Matrix operations test PASSED" << endl;
}

int main() {
    cout << "==================================================" << endl;
    cout << "    Neural Network Unit Tests" << endl;
    cout << "==================================================" << endl;

    try {
        test_sigmoid();
        test_sigmoid_gradient();
        test_element_mult();
        test_forward_pass();
        test_backward_pass();
        test_layer_dependencies();
        test_xor_network();

        cout << endl;
        cout << "==================================================" << endl;
        cout << "    ALL TESTS PASSED!" << endl;
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
