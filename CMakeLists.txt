cmake_minimum_required (VERSION 3.5)
project (NeuralNet)

# set compiler flags
# Use system default compilers instead of g++-5/gcc-5
# set(CMAKE_CXX_COMPILER g++)
# set(CMAKE_C_COMPILER gcc)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native" CACHE STRING "Flags used by the C++ compiler during release builds")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native" CACHE STRING "Flags used by the C compiler during release builds")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "Flags used by the C++ compiler during debug builds")
set(CMAKE_C_FLAGS_DEBUG "-g -O0" CACHE STRING "Flags used by the C compiler during debug builds")

set(CXX_COMMON_FLAGS "" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CXX_COMMON_FLAGS}")
set(CMAKE_C_FLAGS "${CXX_COMMON_FLAGS}")
MESSAGE( STATUS "CMAKE_BUILD_TYPE:      " ${CMAKE_BUILD_TYPE} )
MESSAGE( STATUS "CMAKE_CXX_FLAGS:       " ${CMAKE_CXX_FLAGS} )
MESSAGE( STATUS "CMAKE_C_FLAGS:         " ${CMAKE_CXX_FLAGS} )
add_definitions(-std=c++11 -funroll-loops -fno-permissive)

find_package(OpenMP)
if(OpenMP_FOUND)
    if(OpenMP_C_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    endif()
    if(OpenMP_CXX_FOUND)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DHAVE_OPENMP)
    endif()
    message(STATUS "OpenMP found and enabled")
else()
    message(STATUS "OpenMP not found - building without parallel optimizations")
endif()

set (PROJECT_TARGET_NAME "NeuralNet")

set(NeuralNet_VERSION_MAJOR 1)
set(NeuralNet_VERSION_MINOR 0)

# Options
option(USE_NN_MATRIX "Use Internal Matrix implementation" ON)

# Configure a header file to pass cmake settings
set(RESOURCES_PATH "${PROJECT_SOURCE_DIR}/res/")
configure_file (
  "${PROJECT_SOURCE_DIR}/res/config.h.in"
  "${PROJECT_BINARY_DIR}/res/config.h"
)

# add the binary tree to the search path for include files
include_directories("${PROJECT_BINARY_DIR}")

# include the matrix lib
if (USE_NN_MATRIX)
  include_directories("${PROJECT_SOURCE_DIR}/Matrix")
  add_subdirectory(Matrix)
  set (EXTRA_LIBS ${EXTRA_LIBS} Matrix)
endif(USE_NN_MATRIX)

# include jsonxx for model serialization
include_directories("${PROJECT_SOURCE_DIR}/thirdparty/jsonxx")
add_library(jsonxx STATIC thirdparty/jsonxx/jsonxx.cpp)
set (EXTRA_LIBS ${EXTRA_LIBS} jsonxx)


# add the executable
set (NN_SOURCE_FILES "main.cpp")
add_executable(${PROJECT_TARGET_NAME} ${NN_SOURCE_FILES})
target_link_libraries(${PROJECT_TARGET_NAME} ${EXTRA_LIBS})

# add the install targets
install (TARGETS NeuralNet DESTINATION bin)
install (FILES "${PROJECT_BINARY_DIR}/config.h" DESTINATION include)

# # # #  ENABLE TESTS  # # # #
enable_testing()
add_test(Run NeuralNet)
add_test(Usage NeuralNet -h)
set_tests_properties(Usage
  PROPERTIES
  PASS_REGULAR_EXPRESSION "Usage:.*"
)

# Macro to simplify adding tests
macro (do_test arg result)
  add_test(Comp${arg} NeuralNet ${arg})
  set_tests_properties(Comp${arg}
    PROPERTIES
    PASS_REGULAR_EXPRESSION ${result}
  )
endmacro(do_test)

# # # # TESTS # # # #
# Usage: do_test(arguments, result)
#  ...
# # # #

# Build a CPACK driven installer package
include(InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${NeuralNet_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${NeuralNet_VERSION_MINOR}")
set (CPACK_PACKAGE_CONTACT "oemulrich@gmail.com")
include(CPack)
include(CTest)

# # # # ENABLE UNIT TESTS - RUNTIME # # #

set (TESTS_TARGET_NAME "tests")
set (TESTS_SOURCE_FILES "tests.cpp")
add_executable(${TESTS_TARGET_NAME} ${TESTS_SOURCE_FILES})
target_link_libraries(${TESTS_TARGET_NAME} ${EXTRA_LIBS})
# Drive the post-build command through custom target
add_custom_command(
  OUTPUT test_.lic
  COMMAND Test ${PROJECT_TARGET_NAME}
  DEPENDS tests
  COMMENT "Testing ml/nn"
  VERBATIM
)

add_custom_target(
  RunTests
  DEPENDS test_.lic
)

# # # # NEURAL NETWORK UNIT TESTS # # #

set (NN_TESTS_TARGET_NAME "test_network")
set (NN_TESTS_SOURCE_FILES "test_network.cpp")
add_executable(${NN_TESTS_TARGET_NAME} ${NN_TESTS_SOURCE_FILES})
target_link_libraries(${NN_TESTS_TARGET_NAME} ${EXTRA_LIBS})
add_test(NAME NeuralNetworkTests COMMAND test_network)

# # # # NEURAL NETWORK TRAINING TESTS # # #

set (NN_TRAINING_TESTS_TARGET_NAME "test_training")
set (NN_TRAINING_TESTS_SOURCE_FILES "test_training.cpp")
add_executable(${NN_TRAINING_TESTS_TARGET_NAME} ${NN_TRAINING_TESTS_SOURCE_FILES})
target_link_libraries(${NN_TRAINING_TESTS_TARGET_NAME} ${EXTRA_LIBS})
add_test(NAME NeuralNetworkTrainingTests COMMAND test_training)

# # # # SIMPLE DEBUG TEST # # #

set (SIMPLE_TEST_TARGET_NAME "test_simple")
set (SIMPLE_TEST_SOURCE_FILES "test_simple.cpp")
add_executable(${SIMPLE_TEST_TARGET_NAME} ${SIMPLE_TEST_SOURCE_FILES})
target_link_libraries(${SIMPLE_TEST_TARGET_NAME} ${EXTRA_LIBS})

# # # # XOR MINIMAL TEST # # #

set (XOR_MIN_TEST_TARGET_NAME "test_xor_minimal")
set (XOR_MIN_TEST_SOURCE_FILES "test_xor_minimal.cpp")
add_executable(${XOR_MIN_TEST_TARGET_NAME} ${XOR_MIN_TEST_SOURCE_FILES})
target_link_libraries(${XOR_MIN_TEST_TARGET_NAME} ${EXTRA_LIBS})

# # # # DEBUG WEIGHTS TEST # # #

set (DEBUG_WEIGHTS_TEST_TARGET_NAME "test_debug_weights")
set (DEBUG_WEIGHTS_TEST_SOURCE_FILES "test_debug_weights.cpp")
add_executable(${DEBUG_WEIGHTS_TEST_TARGET_NAME} ${DEBUG_WEIGHTS_TEST_SOURCE_FILES})
target_link_libraries(${DEBUG_WEIGHTS_TEST_TARGET_NAME} ${EXTRA_LIBS})

# # # # COMPREHENSIVE UNIT TESTS # # #

set (COMPREHENSIVE_TEST_TARGET_NAME "test_comprehensive")
set (COMPREHENSIVE_TEST_SOURCE_FILES "test_comprehensive.cpp")
add_executable(${COMPREHENSIVE_TEST_TARGET_NAME} ${COMPREHENSIVE_TEST_SOURCE_FILES})
target_link_libraries(${COMPREHENSIVE_TEST_TARGET_NAME} ${EXTRA_LIBS})
add_test(NAME ComprehensiveTests COMMAND test_comprehensive)

# # # # MODEL SAVE/LOAD TESTS # # #

set (SAVE_LOAD_TEST_TARGET_NAME "test_model_save_load")
set (SAVE_LOAD_TEST_SOURCE_FILES "test_model_save_load.cpp")
add_executable(${SAVE_LOAD_TEST_TARGET_NAME} ${SAVE_LOAD_TEST_SOURCE_FILES})
target_link_libraries(${SAVE_LOAD_TEST_TARGET_NAME} ${EXTRA_LIBS})
add_test(NAME ModelSaveLoadTests COMMAND test_model_save_load)

# # # # MNIST DATA LOADER TESTS # # #

set (MNIST_LOADER_TEST_TARGET_NAME "test_mnist_loader")
set (MNIST_LOADER_TEST_SOURCE_FILES "test_mnist_loader.cpp")
add_executable(${MNIST_LOADER_TEST_TARGET_NAME} ${MNIST_LOADER_TEST_SOURCE_FILES})
target_link_libraries(${MNIST_LOADER_TEST_TARGET_NAME} ${EXTRA_LIBS})
