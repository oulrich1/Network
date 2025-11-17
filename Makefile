.PHONY: run_test_training build_dir test_training

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

