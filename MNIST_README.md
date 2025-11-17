# MNIST Data Loader

This directory contains a complete MNIST data loader implementation for the Neural Network library.

## Overview

The MNIST loader (`mnist_loader.h`) provides functionality to:
- Load MNIST dataset files in IDX format
- Parse and normalize image data (28×28 grayscale images)
- Convert labels to one-hot encoded format
- Access individual samples or batches
- Visualize digits as ASCII art

## Getting the MNIST Dataset

Download the MNIST dataset files from: http://yann.lecun.com/exdb/mnist/

You need these four files:
- `train-images-idx3-ubyte` (9.9 MB) - 60,000 training images
- `train-labels-idx1-ubyte` (29 KB) - 60,000 training labels
- `t10k-images-idx3-ubyte` (1.6 MB) - 10,000 test images
- `t10k-labels-idx1-ubyte` (5 KB) - 10,000 test labels

**Important**: Download the files and place them in your working directory (or adjust the paths in your code).

### Quick Download Commands

```bash
# Download training set
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

# Download test set
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Decompress all files
gunzip *.gz
```

## Usage

### Basic Loading

```cpp
#include "mnist_loader.h"

using namespace ml;

// Load training dataset
MNISTDataset<double> trainDataset;
loadMNISTDataset<double>(
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    trainDataset
);

// Dataset properties
std::cout << "Samples: " << trainDataset.numSamples << std::endl;
std::cout << "Image size: " << trainDataset.imageSize << std::endl;
std::cout << "Classes: " << trainDataset.numClasses << std::endl;
```

### Accessing Individual Samples

```cpp
// Get a single sample (image + label)
auto sample = getSample<double>(trainDataset, 0);
ml::Mat<double> image = sample.first;  // (1, 784) matrix
ml::Mat<double> label = sample.second; // (1, 10) one-hot encoded

// Get the raw label value (0-9)
int rawLabel = trainDataset.rawLabels[0];
```

### Getting Batches for Training

```cpp
// Get a batch of 32 samples starting at index 0
int batchSize = 32;
auto batch = getBatch<double>(trainDataset, 0, batchSize);

ml::Mat<double> batchImages = batch.first;  // (32, 784)
ml::Mat<double> batchLabels = batch.second; // (32, 10)
```

### Visualizing Digits

```cpp
// Print an ASCII visualization of a digit
auto sample = getSample<double>(trainDataset, 0);
printMNISTDigit<double>(sample.first, trainDataset.rawLabels[0]);
```

## Data Format

### Images
- **Dimensions**: Each image is 28×28 pixels = 784 values
- **Format**: Flattened row-major order (rows concatenated)
- **Normalization**: Pixel values are normalized to [0.0, 1.0] range
- **Storage**: `trainDataset.images` is a matrix of size (numSamples, 784)

### Labels
- **Raw Labels**: Integer values 0-9 stored in `trainDataset.rawLabels`
- **One-Hot Encoded**: `trainDataset.labels` is a matrix of size (numSamples, 10)
  - Example: Label "3" becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

## Building and Testing

### Build the MNIST loader test:

```bash
# Using Make
make test_mnist_loader

# Or using CMake directly
cd build
cmake ..
make test_mnist_loader
```

### Run the test:

```bash
# Using Make
make run_test_mnist_loader

# Or directly
cd build
./test_mnist_loader
```

**Note**: Make sure the MNIST dataset files are in the same directory where you run the test.

## Integration with Neural Network

Here's a simple example of training a network on MNIST:

```cpp
#include "network.h"
#include "mnist_loader.h"

using namespace ml;

int main() {
    // Load MNIST data
    MNISTDataset<double> trainDataset;
    loadMNISTDataset<double>(
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        trainDataset
    );

    // Create network: 784 -> 256 -> 128 -> 10
    Network<double>* network = new Network<double>();
    ILayer<double>* input = new Layer<double>(784, "Input", ActivationType::RELU);
    ILayer<double>* hidden1 = new Layer<double>(256, "Hidden1", ActivationType::RELU);
    ILayer<double>* hidden2 = new Layer<double>(128, "Hidden2", ActivationType::RELU);
    ILayer<double>* output = new Layer<double>(10, "Output", ActivationType::SIGMOID);

    network->setInputLayer(input);
    network->connect(input, hidden1);
    network->connect(hidden1, hidden2);
    network->connect(hidden2, output);
    network->setOutputLayer(output);

    // Use Adam optimizer
    network->setOptimizerType(OptimizerType::ADAM);
    network->init();

    // Training loop
    double learningRate = 0.001;
    int epochs = 10;
    int batchSize = 1;  // Single sample per update

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < trainDataset.numSamples; i++) {
            // Get single sample
            auto sample = getSample<double>(trainDataset, i);

            // Forward pass
            ml::Mat<double> predicted = network->feed(sample.first);

            // Compute error
            ml::Mat<double> error = ml::Diff(sample.second, predicted);
            network->getOutputLayer()->setErrors(error);

            // Backward pass
            network->backprop();

            // Update weights
            network->updateWeights(learningRate);
        }

        std::cout << "Epoch " << (epoch + 1) << " complete" << std::endl;
    }

    return 0;
}
```

## File Structure

- `mnist_loader.h` - Main MNIST data loader implementation
- `test_mnist_loader.cpp` - Test suite demonstrating usage
- `MNIST_README.md` - This documentation file

## Technical Details

### IDX File Format

MNIST uses the IDX file format:

**Image File Header (16 bytes)**:
- Bytes 0-3: Magic number (2051 for images)
- Bytes 4-7: Number of images
- Bytes 8-11: Number of rows (28)
- Bytes 12-15: Number of columns (28)
- Remaining: Pixel data (unsigned bytes 0-255)

**Label File Header (8 bytes)**:
- Bytes 0-3: Magic number (2049 for labels)
- Bytes 4-7: Number of labels
- Remaining: Label data (unsigned bytes 0-9)

All integers are stored in MSB (Most Significant Byte) first format (big-endian).

## Next Steps

After successfully loading MNIST data, you can:

1. **Implement a complete MNIST training script** with:
   - Batch training support
   - Training/validation split
   - Accuracy evaluation
   - Model checkpointing

2. **Experiment with different architectures**:
   - Vary hidden layer sizes
   - Try different activation functions
   - Test different optimizers (SGD, Adam, RMSprop)

3. **Add advanced features**:
   - Learning rate scheduling
   - Early stopping
   - Data augmentation
   - Cross-entropy loss for better classification

## Expected Performance

With the current fully-connected architecture (784→256→128→10) and Adam optimizer:
- **Expected accuracy**: 85-92% on test set
- **Training time**: ~5-10 minutes for 10 epochs (CPU-dependent)

For better performance, consider:
- Convolutional layers (when implemented)
- Dropout regularization (when implemented)
- Batch normalization (when implemented)

## Troubleshooting

**Problem**: "Cannot open file" error
- **Solution**: Make sure MNIST files are in the current working directory or provide full paths

**Problem**: "Invalid MNIST file" error
- **Solution**: Ensure files are decompressed (.gz files must be gunzipped)

**Problem**: Segmentation fault or memory errors
- **Solution**: Check that your Matrix library is properly initialized and sized correctly

## References

- MNIST Database: http://yann.lecun.com/exdb/mnist/
- Original Paper: LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)
