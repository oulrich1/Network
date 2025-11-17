#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include "Matrix/matrix.h"

namespace ml {

// Helper function to reverse bytes for big-endian to little-endian conversion
inline uint32_t reverseInt(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

// MNIST Dataset container
template <typename T>
struct MNISTDataset {
    ml::Mat<T> images;      // Each row is a flattened 28x28 image (784 values)
    ml::Mat<T> labels;      // Each row is a one-hot encoded label (10 values)
    std::vector<int> rawLabels; // Original label values (0-9)
    int numSamples;
    int imageSize;          // 784 for MNIST (28x28)
    int numClasses;         // 10 for MNIST (digits 0-9)

    MNISTDataset() : numSamples(0), imageSize(784), numClasses(10) {}
};

/**
 * Read MNIST image file (IDX3-UBYTE format)
 *
 * File format:
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000803(2051) magic number
 * 0004     32 bit integer  60000            number of images
 * 0008     32 bit integer  28               number of rows
 * 0012     32 bit integer  28               number of columns
 * 0016     unsigned byte   ??               pixel
 * 0017     unsigned byte   ??               pixel
 * ........
 * xxxx     unsigned byte   ??               pixel
 */
template <typename T>
bool readMNISTImages(const std::string& filename, ml::Mat<T>& images, int& numImages, int& imageSize) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    // Read magic number
    uint32_t magic = 0;
    file.read((char*)&magic, sizeof(magic));
    magic = reverseInt(magic);
    if (magic != 2051) {
        std::cerr << "Error: Invalid MNIST image file (magic number: " << magic << ")" << std::endl;
        return false;
    }

    // Read dimensions
    uint32_t numImagesU32 = 0, rows = 0, cols = 0;
    file.read((char*)&numImagesU32, sizeof(numImagesU32));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    numImagesU32 = reverseInt(numImagesU32);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    numImages = static_cast<int>(numImagesU32);
    imageSize = rows * cols;

    std::cout << "Loading " << numImages << " images of size "
              << rows << "x" << cols << " = " << imageSize << " pixels" << std::endl;

    // Create matrix: each row is a flattened image
    images = ml::Mat<T>(numImages, imageSize, 0);

    // Read pixel data
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            // Normalize to [0, 1] range
            images.setAt(i, j, static_cast<T>(pixel) / 255.0);
        }
    }

    file.close();
    std::cout << "Successfully loaded " << numImages << " images" << std::endl;
    return true;
}

/**
 * Read MNIST label file (IDX1-UBYTE format)
 *
 * File format:
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 * 0004     32 bit integer  60000            number of items
 * 0008     unsigned byte   ??               label
 * 0009     unsigned byte   ??               label
 * ........
 * xxxx     unsigned byte   ??               label
 */
template <typename T>
bool readMNISTLabels(const std::string& filename, std::vector<int>& rawLabels,
                     ml::Mat<T>& oneHotLabels, int& numLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    // Read magic number
    uint32_t magic = 0;
    file.read((char*)&magic, sizeof(magic));
    magic = reverseInt(magic);
    if (magic != 2049) {
        std::cerr << "Error: Invalid MNIST label file (magic number: " << magic << ")" << std::endl;
        return false;
    }

    // Read number of labels
    uint32_t numLabelsU32 = 0;
    file.read((char*)&numLabelsU32, sizeof(numLabelsU32));
    numLabelsU32 = reverseInt(numLabelsU32);
    numLabels = static_cast<int>(numLabelsU32);

    std::cout << "Loading " << numLabels << " labels" << std::endl;

    // Read labels
    rawLabels.resize(numLabels);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        rawLabels[i] = static_cast<int>(label);
    }

    // Create one-hot encoded labels (10 classes for digits 0-9)
    const int numClasses = 10;
    oneHotLabels = ml::Mat<T>(numLabels, numClasses, 0);

    for (int i = 0; i < numLabels; i++) {
        int label = rawLabels[i];
        if (label >= 0 && label < numClasses) {
            oneHotLabels.setAt(i, label, 1.0);
        }
    }

    file.close();
    std::cout << "Successfully loaded " << numLabels << " labels" << std::endl;
    return true;
}

/**
 * Load MNIST dataset from files
 *
 * @param imageFile Path to MNIST image file (e.g., "train-images-idx3-ubyte")
 * @param labelFile Path to MNIST label file (e.g., "train-labels-idx1-ubyte")
 * @param dataset Output dataset structure
 * @return true if successful, false otherwise
 */
template <typename T>
bool loadMNISTDataset(const std::string& imageFile, const std::string& labelFile,
                      MNISTDataset<T>& dataset) {
    int numImages = 0, imageSize = 0;
    int numLabels = 0;

    // Read images
    if (!readMNISTImages<T>(imageFile, dataset.images, numImages, imageSize)) {
        return false;
    }

    // Read labels
    if (!readMNISTLabels<T>(labelFile, dataset.rawLabels, dataset.labels, numLabels)) {
        return false;
    }

    // Verify consistency
    if (numImages != numLabels) {
        std::cerr << "Error: Number of images (" << numImages
                  << ") doesn't match number of labels (" << numLabels << ")" << std::endl;
        return false;
    }

    dataset.numSamples = numImages;
    dataset.imageSize = imageSize;
    dataset.numClasses = 10;

    std::cout << "MNIST dataset loaded successfully:" << std::endl;
    std::cout << "  - Samples: " << dataset.numSamples << std::endl;
    std::cout << "  - Image size: " << dataset.imageSize << " pixels" << std::endl;
    std::cout << "  - Classes: " << dataset.numClasses << std::endl;

    return true;
}

/**
 * Helper function to get a single training sample
 * Returns a pair of (image, label) matrices, each as a single row
 */
template <typename T>
std::pair<ml::Mat<T>, ml::Mat<T>> getSample(const MNISTDataset<T>& dataset, int index) {
    if (index < 0 || index >= dataset.numSamples) {
        throw std::out_of_range("Sample index out of range");
    }

    // Extract single row for image and label
    ml::Mat<T> image(1, dataset.imageSize, 0);
    ml::Mat<T> label(1, dataset.numClasses, 0);

    for (int i = 0; i < dataset.imageSize; i++) {
        image.setAt(0, i, dataset.images.getAt(index, i));
    }

    for (int i = 0; i < dataset.numClasses; i++) {
        label.setAt(0, i, dataset.labels.getAt(index, i));
    }

    return std::make_pair(image, label);
}

/**
 * Helper function to get a batch of samples
 * Returns a pair of (images, labels) matrices
 */
template <typename T>
std::pair<ml::Mat<T>, ml::Mat<T>> getBatch(const MNISTDataset<T>& dataset,
                                           int startIdx, int batchSize) {
    if (startIdx < 0 || startIdx >= dataset.numSamples) {
        throw std::out_of_range("Start index out of range");
    }

    // Clamp batch size to available samples
    int actualBatchSize = std::min(batchSize, dataset.numSamples - startIdx);

    ml::Mat<T> images(actualBatchSize, dataset.imageSize, 0);
    ml::Mat<T> labels(actualBatchSize, dataset.numClasses, 0);

    for (int i = 0; i < actualBatchSize; i++) {
        int srcIdx = startIdx + i;
        for (int j = 0; j < dataset.imageSize; j++) {
            images.setAt(i, j, dataset.images.getAt(srcIdx, j));
        }
        for (int j = 0; j < dataset.numClasses; j++) {
            labels.setAt(i, j, dataset.labels.getAt(srcIdx, j));
        }
    }

    return std::make_pair(images, labels);
}

/**
 * Print ASCII visualization of an MNIST digit
 */
template <typename T>
void printMNISTDigit(const ml::Mat<T>& image, int label) {
    std::cout << "Label: " << label << std::endl;

    // Assume image is either a single row vector (1, 784) or already the pixel values
    int numPixels = image.size().cx;
    if (numPixels != 784) {
        std::cerr << "Error: Image must have 784 pixels" << std::endl;
        return;
    }

    const char* grayscale = " .:-=+*#%@";
    const int levels = 10;

    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            int idx = row * 28 + col;
            T pixelValue = image.getAt(0, idx);
            int level = static_cast<int>(pixelValue * (levels - 1));
            level = std::min(std::max(level, 0), levels - 1);
            std::cout << grayscale[level] << grayscale[level];
        }
        std::cout << std::endl;
    }
}

} // namespace ml
