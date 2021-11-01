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

cd python
# This is for changelog
git fetch origin master
TAICHI_CMAKE_ARGS=$CI_SETUP_CMAKE_ARGS python3 build.py build --project_name $PEOJECT_NAME
# Run basic cpp tests
cd ..
CUR_DIR=`pwd`
TI_LIB_DIR=$CUR_DIR/python/taichi/lib ./build/taichi_cpp_tests

cp dist/*.whl /wheel/
