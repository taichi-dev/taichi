#!/bin/bash

set -ex

# Parse ARGs
PY=$1
GPU_BUILD=$2
PROJECT_NAME=$3
CI_SETUP_CMAKE_ARGS=$4
export SCCACHE_DIR=/cache
export SCCACHE_CACHE_SIZE="256M"
source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

python3 -m pip uninstall taichi taichi-nightly -y
wget https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz
tar -xzf sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz
chmod +x sccache-v0.2.15-x86_64-unknown-linux-musl/sccache
export PATH=$(pwd)/sccache-v0.2.15-x86_64-unknown-linux-musl:$PATH
sccache --start-server
cd taichi

if [[ $GPU_BUILD == "OFF" ]]
then 
    python3 -m pip install -r requirements_dev.txt
fi

cd python
# This is for changelog
git fetch origin master
TAICHI_CMAKE_ARGS=$CI_SETUP_CMAKE_ARGS python3 build.py build --project_name $PROJECT_NAME
# Run basic cpp tests
cd ..
CUR_DIR=`pwd`
TI_LIB_DIR=$CUR_DIR/python/taichi/lib ./build/taichi_cpp_tests
sccache -s
cp dist/*.whl /wheel/
