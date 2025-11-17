.PHONY: run_test_training build_dir test_training test_mnist_loader run_test_mnist_loader train_mnist run_mnist test_mnist_training run_test_mnist_training

# Default target
all: build_dir
	cd build && make

# Ensure build directory exists and cmake has been run
build_dir:
	@if [ ! -d "./build" ]; then \
		mkdir -p build; \
	fi
	@if [ ! -f "./build/Makefile" ]; then \
		cd build && cmake ..; \
	fi

# Build the test_training target
test_training: build_dir
	cd build && make test_training

# Build and run test_training
run_test_training: test_training
	cd build && ./test_training

# Build the test_mnist_loader target
test_mnist_loader: build_dir
	cd build && make test_mnist_loader

# Build and run test_mnist_loader
run_test_mnist_loader: test_mnist_loader
	cd build && ./test_mnist_loader

# Build the train_mnist target
train_mnist: build_dir
	cd build && make train_mnist

# Build and run train_mnist
run_mnist: train_mnist
	cd build && ./train_mnist

# Build the test_mnist_training target
test_mnist_training: build_dir
	cd build && make test_mnist_training

# Build and run test_mnist_training
run_test_mnist_training: test_mnist_training
	cd build && ./test_mnist_training

