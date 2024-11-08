cmake_minimum_required(VERSION 3.10)

# Project name and programming language
project(Tacc LANGUAGES CXX CUDA)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set GPU Computing Capability
set(CMAKE_CUDA_ARCHITECTURES 86)
# Specify CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)
# Add the executable with cpp or cu file
add_executable(Tacc 
    lib/util.cu
    src/main.cu)

target_link_libraries(Tacc PRIVATE cudart cublas)

enable_testing()
add_subdirectory(test)
add_subdirectory(googletest)