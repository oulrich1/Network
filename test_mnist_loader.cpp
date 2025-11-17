#include <iostream>
#include <stdlib.h>
#include "allheader.h"
#include "mnist_loader.h"

using namespace std;
using namespace ml;
using namespace Utility;

/**
 * Test MNIST data loading functionality
 *
 * This test demonstrates how to:
 * 1. Load MNIST dataset from IDX files
 * 2. Access individual samples
 * 3. Visualize digits
 * 4. Get batches of data for training
 *
 * To run this test, you need MNIST dataset files:
 *   - train-images-idx3-ubyte
 *   - train-labels-idx1-ubyte
 *   - t10k-images-idx3-ubyte (optional, for test set)
 *   - t10k-labels-idx1-ubyte (optional, for test set)
 *
 * Download from: http://yann.lecun.com/exdb/mnist/
 */
void test_mnist_loading() {
    BEGIN_TESTS("Testing MNIST Data Loading");
    typedef double T;

    // Path to MNIST data files (adjust these paths as needed)
    const string trainImagesFile = "train-images-idx3-ubyte";
    const string trainLabelsFile = "train-labels-idx1-ubyte";

    // Load training dataset
    cout << "\n=== Loading MNIST Training Dataset ===" << endl;
    MNISTDataset<T> trainDataset;

    if (!loadMNISTDataset<T>(trainImagesFile, trainLabelsFile, trainDataset)) {
        cout << "Failed to load MNIST dataset. Make sure the MNIST files are in the current directory." << endl;
        cout << "Download MNIST from: http://yann.lecun.com/exdb/mnist/" << endl;
        return;
    }

    // Verify dataset properties
    cout << "\n=== Dataset Properties ===" << endl;
    cout << "Number of training samples: " << trainDataset.numSamples << endl;
    cout << "Image dimensions: 28x28 = " << trainDataset.imageSize << " pixels" << endl;
    cout << "Number of classes: " << trainDataset.numClasses << endl;
    cout << "Images matrix size: (" << trainDataset.images.size().cy
         << ", " << trainDataset.images.size().cx << ")" << endl;
    cout << "Labels matrix size: (" << trainDataset.labels.size().cy
         << ", " << trainDataset.labels.size().cx << ")" << endl;

    // Display first few samples
    cout << "\n=== Displaying Sample Digits ===" << endl;
    for (int i = 0; i < 3; i++) {
        auto sample = getSample<T>(trainDataset, i);
        cout << "\nSample " << i << ":" << endl;
        printMNISTDigit<T>(sample.first, trainDataset.rawLabels[i]);

        // Show the one-hot encoded label
        cout << "One-hot label: [";
        for (int j = 0; j < trainDataset.numClasses; j++) {
            cout << sample.second.getAt(0, j);
            if (j < trainDataset.numClasses - 1) cout << ", ";
        }
        cout << "]" << endl;
    }

    // Test batch loading
    cout << "\n=== Testing Batch Loading ===" << endl;
    int batchSize = 32;
    auto batch = getBatch<T>(trainDataset, 0, batchSize);

    cout << "Batch images matrix: (" << batch.first.size().cy
         << ", " << batch.first.size().cx << ")" << endl;
    cout << "Batch labels matrix: (" << batch.second.size().cy
         << ", " << batch.second.size().cx << ")" << endl;

    // Verify pixel value ranges (should be normalized to [0, 1])
    cout << "\n=== Verifying Data Normalization ===" << endl;
    T minVal = 1.0, maxVal = 0.0;
    for (int i = 0; i < trainDataset.imageSize; i++) {
        T val = trainDataset.images.getAt(0, i);
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
    }
    cout << "Pixel value range: [" << minVal << ", " << maxVal << "]" << endl;

    // Show label distribution
    cout << "\n=== Label Distribution (first 1000 samples) ===" << endl;
    int labelCounts[10] = {0};
    int samplesToCheck = std::min(1000, trainDataset.numSamples);
    for (int i = 0; i < samplesToCheck; i++) {
        int label = trainDataset.rawLabels[i];
        if (label >= 0 && label < 10) {
            labelCounts[label]++;
        }
    }
    for (int i = 0; i < 10; i++) {
        cout << "Digit " << i << ": " << labelCounts[i] << " samples" << endl;
    }

    cout << "\n=== MNIST Loading Test Complete ===" << endl;
}

/**
 * Test loading MNIST test set (optional)
 */
void test_mnist_test_set() {
    BEGIN_TESTS("Testing MNIST Test Set Loading");
    typedef double T;

    const string testImagesFile = "t10k-images-idx3-ubyte";
    const string testLabelsFile = "t10k-labels-idx1-ubyte";

    cout << "\n=== Loading MNIST Test Dataset ===" << endl;
    MNISTDataset<T> testDataset;

    if (!loadMNISTDataset<T>(testImagesFile, testLabelsFile, testDataset)) {
        cout << "Failed to load MNIST test dataset." << endl;
        return;
    }

    cout << "Test set size: " << testDataset.numSamples << " samples" << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "    MNIST Data Loader Test Suite       " << endl;
    cout << "========================================" << endl;

    test_mnist_loading();

    // Optionally test the test set
    cout << "\n\n";
    test_mnist_test_set();

    return 0;
}
