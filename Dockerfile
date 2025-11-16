# Neural Network Project - Docker Build Environment
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    gcc \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Create build directory and build the project
RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    make

# Run tests to verify the build
RUN cd build && \
    ./test_network && \
    echo "Build and tests completed successfully!"

# Default command
CMD ["./build/NeuralNet"]
