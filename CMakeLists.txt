# CMake version requirement
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(Tacc)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)
# Add the executable 
add_executable(Tacc
    src/main.cpp)

enable_testing()
add_subdirectory(test)
add_subdirectory(googletest)