#!/bin/bash

set -ex

# Parse ARGs
PY=$1
GPU_BUILD=$2
CI_SETUP_CMAKE_ARGS=$3

chown -R dev taichi
ls -al

source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

cd taichi
ls -al
[[ $GPU_BUILD == "OFF" ]] && python3 -m pip install -r requirements_dev.txt
cd python
TAICHI_CMAKE_ARGS=$CI_SETUP_CMAKE_ARGS python3 build.py build
# Run basic cpp tests
cd ..
CUR_DIR=`pwd`
TI_LIB_DIR=$CUR_DIR/python/taichi/lib ./build/taichi_cpp_tests

tail -f /dev/null
