#!/bin/sh

set -x

### Path to clang++ ###
# export CXX=/path/to/clang++

### LLVM toolchain directory ###
# export LLVM_DIR=/usr/local/lib/cmake/llvm

### Additional CMake args for building Taichi ###
# export TAICHI_CMAKE_ARGS="-DCMAKE_CXX_COMPILER=clang++"

### Number of threads used when running Taichi tests ###
# export TI_TEST_THREADS=4

set +x
