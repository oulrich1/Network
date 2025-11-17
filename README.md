# Neural Network Library

A C++ implementation of a basic neural network with forward and backward propagation, featuring flexible layer architecture and comprehensive unit tests.

## Features

- ✅ Forward propagation with sigmoid activation
- ✅ Backward propagation for error computation
- ✅ Flexible layer architecture (supports arbitrary network topologies)
- ✅ Matrix operations optimized with OpenMP
- ✅ Model serialization (save/load weights to/from JSON files)
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

# Run model save/load tests
make test_model_save_load
./test_model_save_load

# Or use CTest to run all tests
ctest --output-on-failure
```

## Running Toy Example Training

The project includes toy example training tests that demonstrate network learning on simple patterns:

### Quick Start

```bash
# From project root
./build.sh                     # Build the project
cd build
make test_training             # Compile training tests
./test_training                # Run all training tests
```

### What Gets Tested

The `test_training` executable runs multiple training scenarios:

1. **Linear Regression** (`y = 2x + 1`)
   - Architecture: 1-8-1 (1 input, 8 hidden, 1 output)
   - 50,000 epochs with learning rate 0.1
   - Target: MSE < 0.01
   - Performance: ~5-15 seconds (with -O3 optimization)

2. **XOR Logic Gate**
   - Architecture: 2-4-1
   - 10,000 epochs with learning rate 0.1
   - Target: >90% accuracy

3. **AND Logic Gate**
   - Architecture: 2-2-1
   - 5,000 epochs with learning rate 0.5
   - Target: >90% accuracy

4. **OR Logic Gate**
   - Architecture: 2-2-1
   - 5,000 epochs with learning rate 0.5
   - Target: >90% accuracy

### Performance Timing

Each test now includes timing information:
- **Training time**: Time spent in the training loop
- **Testing time**: Time spent evaluating the network

Example output:
```
>> Training complete. Time: 12345 ms
>> Testing complete. Time: 2 ms
>> Final MSE (on normalized values): 0.00173
>> Accuracy (within 5% tolerance): 72.7% (8/11 samples)
```

### Building with Optimizations

For faster training, build in Release mode with optimizations:

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make test_training -j4
./test_training
```

**Performance difference:**
- Debug mode (`-O0`): ~5+ minutes for linear regression
- Release mode (`-O3`): ~5-15 seconds for linear regression (10-60x faster!)

### Comparing with PyTorch

A PyTorch baseline implementation is provided for comparison:

```bash
# Install dependencies (if needed)
pip install torch numpy matplotlib

# Run PyTorch comparison
python3 pytorch_toy_example/linear_regression_comparison.py
```

This generates:
- Console output with training progress
- Comparison plot: `pytorch_toy_example/comparison_results.png`
- Performance benchmarks for both linear regression and neural network approaches

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

## Model Serialization

Save and load trained network weights to/from JSON files:

### Saving a Model

```cpp
// After training your network...
network->init();
// ... perform training ...

// Save the model to a file
if (network->saveToFile("my_model.json")) {
    std::cout << "Model saved successfully!" << std::endl;
}
```

### Loading a Model

```cpp
// Create a network with the same structure as the saved model
Network<double>* network = new Network<double>();

ILayer<double>* inputLayer  = new Layer<double>(2, "Input");
ILayer<double>* hiddenLayer = new Layer<double>(4, "Hidden");
ILayer<double>* outputLayer = new Layer<double>(1, "Output");

network->setInputLayer(inputLayer);
network->connect(inputLayer, hiddenLayer);
network->connect(hiddenLayer, outputLayer);
network->setOutputLayer(outputLayer);

// Initialize with random weights (will be overwritten)
network->init();

// Load the saved weights
if (network->loadFromFile("my_model.json")) {
    std::cout << "Model loaded successfully!" << std::endl;

    // Now you can use the network for inference
    Mat<double> input(1, 2, 0);
    input.setAt(0, 0, 1.0);
    input.setAt(0, 1, 0.5);
    Mat<double> output = network->feed(input);
}
```

### JSON File Format

The saved model contains:
- **layers**: Array of layer information (name, input size, output size)
- **weights**: Weight matrices between connected layers

Example:
```json
{
  "layers": [
    {"index": 0, "name": "Input", "input_size": 2, "output_size": 3},
    {"index": 1, "name": "Hidden", "input_size": 4, "output_size": 5},
    {"index": 2, "name": "Output", "input_size": 1, "output_size": 2}
  ],
  "weights": [
    {
      "from_index": 0,
      "to_index": 1,
      "rows": 4,
      "cols": 3,
      "values": [0.123, -0.456, ...]
    }
  ]
}
```

**Important**: When loading a model, the network structure (number of layers and their sizes) must match the saved model exactly.

### Testing Save/Load

Run the save/load test to see a complete example:

```bash
cd build
make test_model_save_load
./test_model_save_load
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

### Platform Support

The library supports both x86/x64 and ARM architectures (including Apple Silicon). SSE optimizations are automatically disabled on non-x86 platforms. The build system will automatically configure the appropriate settings for your platform.

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass: `make test_network && ./test_network`
2. Code follows existing style conventions
3. New features include corresponding unit tests

## License

See `License.txt` for details.
