#include <iostream>
#include <cassert>
#include <cmath>
#include "network.h"

using namespace ml;
using namespace std;

// Helper function to check if two values are approximately equal
template<typename T>
bool approxEqual(T a, T b, T epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

// Test activation functions
void testActivationFunctions() {
    cout << "\n=== Testing Activation Functions ===" << endl;

    // Test Sigmoid
    {
        Mat<double> input(1, 3, 0);
        input.setAt(0, 0, -1.0);
        input.setAt(0, 1, 0.0);
        input.setAt(0, 2, 1.0);

        Mat<double> output = Sigmoid(input);
        assert(approxEqual(output.getAt(0, 0), 0.2689414, 1e-5));
        assert(approxEqual(output.getAt(0, 1), 0.5, 1e-5));
        assert(approxEqual(output.getAt(0, 2), 0.7310586, 1e-5));

        Mat<double> grad = SigmoidGrad(output);
        assert(approxEqual(grad.getAt(0, 0), 0.1966119, 1e-5));
        assert(approxEqual(grad.getAt(0, 1), 0.25, 1e-5));
        assert(approxEqual(grad.getAt(0, 2), 0.1966119, 1e-5));

        cout << "✓ Sigmoid activation and gradient test passed" << endl;
    }

    // Test ReLU
    {
        Mat<double> input(1, 4, 0);
        input.setAt(0, 0, -2.0);
        input.setAt(0, 1, -0.5);
        input.setAt(0, 2, 0.0);
        input.setAt(0, 3, 1.5);

        Mat<double> output = ReLU(input);
        assert(approxEqual(output.getAt(0, 0), 0.0));
        assert(approxEqual(output.getAt(0, 1), 0.0));
        assert(approxEqual(output.getAt(0, 2), 0.0));
        assert(approxEqual(output.getAt(0, 3), 1.5));

        Mat<double> grad = ReLUGrad(output);
        assert(approxEqual(grad.getAt(0, 0), 0.0));
        assert(approxEqual(grad.getAt(0, 1), 0.0));
        assert(approxEqual(grad.getAt(0, 2), 0.0));
        assert(approxEqual(grad.getAt(0, 3), 1.0));

        cout << "✓ ReLU activation and gradient test passed" << endl;
    }

    // Test Leaky ReLU
    {
        Mat<double> input(1, 3, 0);
        input.setAt(0, 0, -1.0);
        input.setAt(0, 1, 0.0);
        input.setAt(0, 2, 1.0);

        Mat<double> output = LeakyReLU(input, 0.01);
        assert(approxEqual(output.getAt(0, 0), -0.01));
        assert(approxEqual(output.getAt(0, 1), 0.0));
        assert(approxEqual(output.getAt(0, 2), 1.0));

        Mat<double> grad = LeakyReLUGrad(output, 0.01);
        assert(approxEqual(grad.getAt(0, 0), 0.01));
        assert(approxEqual(grad.getAt(0, 1), 0.01));  // gradient is alpha at x=0
        assert(approxEqual(grad.getAt(0, 2), 1.0));

        cout << "✓ Leaky ReLU activation and gradient test passed" << endl;
    }

    // Test Tanh
    {
        Mat<double> input(1, 3, 0);
        input.setAt(0, 0, -1.0);
        input.setAt(0, 1, 0.0);
        input.setAt(0, 2, 1.0);

        Mat<double> output = Tanh(input);
        assert(approxEqual(output.getAt(0, 0), -0.7615942, 1e-5));
        assert(approxEqual(output.getAt(0, 1), 0.0));
        assert(approxEqual(output.getAt(0, 2), 0.7615942, 1e-5));

        Mat<double> grad = TanhGrad(output);
        assert(approxEqual(grad.getAt(0, 0), 0.4199743, 1e-5));
        assert(approxEqual(grad.getAt(0, 1), 1.0));
        assert(approxEqual(grad.getAt(0, 2), 0.4199743, 1e-5));

        cout << "✓ Tanh activation and gradient test passed" << endl;
    }

    // Test Softmax
    {
        Mat<double> input(1, 3, 0);
        input.setAt(0, 0, 1.0);
        input.setAt(0, 1, 2.0);
        input.setAt(0, 2, 3.0);

        Mat<double> output = Softmax(input);
        double sum = output.getAt(0, 0) + output.getAt(0, 1) + output.getAt(0, 2);
        assert(approxEqual(sum, 1.0, 1e-5));
        assert(approxEqual(output.getAt(0, 0), 0.0900306, 1e-5));
        assert(approxEqual(output.getAt(0, 1), 0.2447285, 1e-5));
        assert(approxEqual(output.getAt(0, 2), 0.6652409, 1e-5));

        cout << "✓ Softmax activation test passed" << endl;
    }

    // Test Linear
    {
        Mat<double> input(1, 3, 0);
        input.setAt(0, 0, -1.5);
        input.setAt(0, 1, 0.0);
        input.setAt(0, 2, 2.5);

        Mat<double> output = Linear(input);
        assert(approxEqual(output.getAt(0, 0), -1.5));
        assert(approxEqual(output.getAt(0, 1), 0.0));
        assert(approxEqual(output.getAt(0, 2), 2.5));

        Mat<double> grad = LinearGrad(output);
        assert(approxEqual(grad.getAt(0, 0), 1.0));
        assert(approxEqual(grad.getAt(0, 1), 1.0));
        assert(approxEqual(grad.getAt(0, 2), 1.0));

        cout << "✓ Linear activation and gradient test passed" << endl;
    }

    // Test unified interface
    {
        Mat<double> input(1, 2, 0);
        input.setAt(0, 0, -1.0);
        input.setAt(0, 1, 1.0);

        Mat<double> sigmoidOut = Activate(input, ActivationType::SIGMOID);
        Mat<double> reluOut = Activate(input, ActivationType::RELU);
        Mat<double> tanhOut = Activate(input, ActivationType::TANH);

        assert(sigmoidOut.getAt(0, 0) > 0.0 && sigmoidOut.getAt(0, 0) < 1.0);
        assert(reluOut.getAt(0, 0) == 0.0 && reluOut.getAt(0, 1) == 1.0);
        assert(tanhOut.getAt(0, 0) < 0.0 && tanhOut.getAt(0, 1) > 0.0);

        cout << "✓ Unified activation interface test passed" << endl;
    }

    cout << "\n✅ All activation function tests passed!\n" << endl;
}

// Test optimizers
void testOptimizers() {
    cout << "\n=== Testing Optimizers ===" << endl;

    // Test SGD
    {
        SGDOptimizer<double> sgd;
        Mat<double> weights(2, 2, 0);
        weights.setAt(0, 0, 1.0);
        weights.setAt(0, 1, 2.0);
        weights.setAt(1, 0, 3.0);
        weights.setAt(1, 1, 4.0);

        Mat<double> gradients(2, 2, 0);
        gradients.setAt(0, 0, 0.1);
        gradients.setAt(0, 1, 0.2);
        gradients.setAt(1, 0, 0.3);
        gradients.setAt(1, 1, 0.4);

        sgd.updateWeights(weights, gradients, 0.1, "test_layer");

        // SGD: w = w + lr * grad
        assert(approxEqual(weights.getAt(0, 0), 1.01));
        assert(approxEqual(weights.getAt(0, 1), 2.02));
        assert(approxEqual(weights.getAt(1, 0), 3.03));
        assert(approxEqual(weights.getAt(1, 1), 4.04));

        cout << "✓ SGD optimizer test passed" << endl;
    }

    // Test Momentum
    {
        MomentumOptimizer<double> momentum(0.9);
        Mat<double> weights(2, 2, 0);
        weights.setAt(0, 0, 1.0);
        weights.setAt(0, 1, 2.0);
        weights.setAt(1, 0, 3.0);
        weights.setAt(1, 1, 4.0);

        Mat<double> gradients(2, 2, 0);
        gradients.setAt(0, 0, 0.1);
        gradients.setAt(0, 1, 0.2);
        gradients.setAt(1, 0, 0.3);
        gradients.setAt(1, 1, 0.4);

        // First update: v = 0*0.9 + grad = grad, w = w + lr * v
        momentum.updateWeights(weights, gradients, 0.1, "test_layer");
        assert(approxEqual(weights.getAt(0, 0), 1.01));
        assert(approxEqual(weights.getAt(0, 1), 2.02));

        // Second update: v = prev_v*0.9 + grad, w = w + lr * v
        momentum.updateWeights(weights, gradients, 0.1, "test_layer");
        // v[0,0] = 0.1*0.9 + 0.1 = 0.19, w[0,0] = 1.01 + 0.1*0.19 = 1.029
        assert(approxEqual(weights.getAt(0, 0), 1.029, 1e-5));

        cout << "✓ Momentum optimizer test passed" << endl;
    }

    // Test Adam
    {
        AdamOptimizer<double> adam(0.9, 0.999, 1e-8);
        Mat<double> weights(2, 2, 0);
        weights.setAt(0, 0, 1.0);
        weights.setAt(0, 1, 2.0);
        weights.setAt(1, 0, 3.0);
        weights.setAt(1, 1, 4.0);

        Mat<double> gradients(2, 2, 0);
        gradients.setAt(0, 0, 0.1);
        gradients.setAt(0, 1, 0.2);
        gradients.setAt(1, 0, 0.3);
        gradients.setAt(1, 1, 0.4);

        double w00_before = weights.getAt(0, 0);
        adam.updateWeights(weights, gradients, 0.01, "test_layer");
        double w00_after = weights.getAt(0, 0);

        // Adam should update weights (exact value depends on bias correction)
        assert(w00_after > w00_before);

        // Second update should also work
        adam.updateWeights(weights, gradients, 0.01, "test_layer");
        assert(weights.getAt(0, 0) > w00_after);

        cout << "✓ Adam optimizer test passed" << endl;
    }

    // Test optimizer reset
    {
        MomentumOptimizer<double> momentum(0.9);
        Mat<double> weights(2, 2, 1.0);
        Mat<double> gradients(2, 2, 0.1);

        momentum.updateWeights(weights, gradients, 0.1, "test_layer");
        double w1 = weights.getAt(0, 0);

        momentum.updateWeights(weights, gradients, 0.1, "test_layer");
        double w2 = weights.getAt(0, 0);

        // Reset and update again
        momentum.reset();
        weights.setAt(0, 0, 1.0); // Reset weight
        momentum.updateWeights(weights, gradients, 0.1, "test_layer");
        double w3 = weights.getAt(0, 0);

        // After reset, first update should be same as initial first update
        assert(approxEqual(w1, w3));

        cout << "✓ Optimizer reset test passed" << endl;
    }

    cout << "\n✅ All optimizer tests passed!\n" << endl;
}

// Test network with different activations
void testNetworkWithActivations() {
    cout << "\n=== Testing Network with Different Activations ===" << endl;

    // Create a simple network with ReLU activation
    Layer<double>* input = new Layer<double>(2, "input", ActivationType::RELU);
    Layer<double>* hidden = new Layer<double>(3, "hidden", ActivationType::RELU);
    Layer<double>* output = new Layer<double>(1, "output", ActivationType::SIGMOID);

    Network<double> net;
    net.connect(input, hidden);
    net.connect(hidden, output);
    net.setInputLayer(input);
    net.setOutputLayer(output);
    net.init();

    // Test feed forward
    Mat<double> testInput(1, 2, 0);
    testInput.setAt(0, 0, 0.5);
    testInput.setAt(0, 1, 0.3);

    Mat<double> result = net.feed(testInput);
    assert(result.IsGood());
    assert(result.size().cx == 1);  // Output should be 1D

    cout << "✓ Network with ReLU activation test passed" << endl;

    // Test with Adam optimizer
    net.setOptimizerType(OptimizerType::ADAM);
    assert(net.getOptimizer()->getType() == OptimizerType::ADAM);

    Mat<double> target(1, 1, 0);
    target.setAt(0, 0, 1.0);

    result = net.feed(testInput);
    output->setErrors(ml::Diff(target, result));
    net.backprop();
    net.updateWeights(0.01);

    cout << "✓ Network with Adam optimizer test passed" << endl;

    // Test with Momentum optimizer
    net.setOptimizerType(OptimizerType::MOMENTUM);
    assert(net.getOptimizer()->getType() == OptimizerType::MOMENTUM);

    result = net.feed(testInput);
    output->setErrors(ml::Diff(target, result));
    net.backprop();
    net.updateWeights(0.01);

    cout << "✓ Network with Momentum optimizer test passed" << endl;

    cout << "\n✅ All network integration tests passed!\n" << endl;
}

int main() {
    cout << "\n========================================" << endl;
    cout << "Testing Activation Functions & Optimizers" << endl;
    cout << "========================================" << endl;

    try {
        testActivationFunctions();
        testOptimizers();
        testNetworkWithActivations();

        cout << "\n========================================" << endl;
        cout << "🎉 ALL TESTS PASSED! 🎉" << endl;
        cout << "========================================\n" << endl;
        return 0;
    } catch (const std::exception& e) {
        cout << "\n❌ TEST FAILED: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "\n❌ TEST FAILED: Unknown error" << endl;
        return 1;
    }
}
