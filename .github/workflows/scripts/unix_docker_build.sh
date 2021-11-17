#!/bin/bash

set -ex

# Parse ARGs
PY=$1
GPU_BUILD=$2
PROJECT_NAME=$3
CI_SETUP_CMAKE_ARGS=$4

source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

python3 -m pip uninstall taichi taichi-nightly -y

cd taichi

if [[ $GPU_BUILD == "OFF" ]]
then
    python3 -m pip install -r requirements_dev.txt
fi

# This is for changelog
git fetch origin master
TAICHI_CMAKE_ARGS=$CI_SETUP_CMAKE_ARGS PROJECT_NAME=$PROJECT_NAME python3 setup.py bdist_wheel
# Run basic cpp tests

CUR_DIR=`pwd`
TI_LIB_DIR=$CUR_DIR/python/taichi/lib ./build/taichi_cpp_tests

cp dist/*.whl /wheel/
