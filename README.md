# Neural Network Library

A C++ implementation of a basic neural network with forward and backward propagation, featuring flexible layer architecture and comprehensive unit tests.

## Features

- ✅ Forward propagation with sigmoid activation
- ✅ Backward propagation for error computation
- ✅ Flexible layer architecture (supports arbitrary network topologies)
- ✅ Matrix operations optimized with OpenMP
- ✅ Comprehensive unit test suite
- ✅ CI/CD with GitHub Actions

## Building

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Running Tests

```bash
# Build and run unit tests
cd build
make test_network
./test_network

# Or use CTest
ctest -R NeuralNetworkTests --output-on-failure
```

## Usage Example

```cpp
#include "network.h"

using namespace ml;

// Create a simple 3-layer network: 2 inputs -> 4 hidden -> 1 output
Network<double>* network = new Network<double>();

ILayer<double>* inputLayer  = new Layer<double>(2, "Input");
ILayer<double>* hiddenLayer = new Layer<double>(4, "Hidden");
ILayer<double>* outputLayer = new Layer<double>(1, "Output");

// Connect layers
network->setInputLayer(inputLayer);
network->connect(inputLayer, hiddenLayer);
network->connect(hiddenLayer, outputLayer);
network->setOutputLayer(outputLayer);

// Initialize weights
network->init();

// Forward pass
Mat<double> input(1, 2, 0);
input.setAt(0, 0, 1.0);
input.setAt(0, 1, 0.5);

Mat<double> output = network->feed(input);

// Backward pass
Mat<double> targetOutput(1, 1, 0.8);
Mat<double> error = Diff(targetOutput, output);
outputLayer->setErrors(error);
network->backprop();

// Access propagated errors
Mat<double> hiddenErrors = hiddenLayer->getErrors();
Mat<double> inputErrors = inputLayer->getErrors();
```

## Matrix Operations

The library includes optimized matrix operations:

```cpp
// Element-wise multiplication
Mat<double> result = ElementMult(m1, m2);

// Matrix multiplication
Mat<double> result = Mult(m1, m2);

// Sigmoid activation
Mat<double> activated = Sigmoid(input);

// Sigmoid gradient
Mat<double> grad = SigGrad(activated);
```

## Architecture

- **ILayer**: Base interface for all layer types
- **Layer**: Standard neural network layer implementation
- **Network**: Container that manages multiple layers and propagation
- **Mat<T>**: Template-based matrix class with reference counting

## Requirements

- C++11 or later
- CMake 3.2+
- OpenMP (optional, for parallel operations)
- GCC/Clang with C++ support

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass: `make test_network && ./test_network`
2. Code follows existing style conventions
3. New features include corresponding unit tests

## License

See `License.txt` for details.
