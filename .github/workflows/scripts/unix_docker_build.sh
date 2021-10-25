#!/bin/bash

set -ex

# Parse ARGs
PY=$1
CI_SETUP_CMAKE_ARGS=$2

source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

cd taichi
TAICHI_CMAKE_ARGS=$CI_SETUP_CMAKE_ARGS python3 build.py build
# Run basic cpp tests
TI_LIB_DIR="$TI_LIB_DIR/lib" ./build/taichi_cpp_tests
