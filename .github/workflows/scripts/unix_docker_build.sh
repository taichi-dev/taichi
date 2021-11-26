#!/bin/bash

set -ex

# Parse ARGs
PY=$1
GPU_BUILD=$2
PROJECT_NAME=$3
CI_SETUP_CMAKE_ARGS=$4
export SCCACHE_DIR=/cache
export SCCACHE_CACHE_SIZE="128M"
export SCCACHE_LOG=debug
export SCCACHE_ERROR_LOG=/home/dev/sccache_error
source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

python3 -m pip uninstall taichi taichi-nightly -y
wget https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz
tar -xzf sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz
chmod +x sccache-v0.2.15-x86_64-unknown-linux-musl/sccache
export PATH=$(pwd)/sccache-v0.2.15-x86_64-unknown-linux-musl:$PATH
sccache -s

cd taichi
git fetch origin master

if [[ $GPU_BUILD == "OFF" ]]
then
    python3 -m pip install -r requirements_dev.txt
fi

PROJECT_TAGS=""
EXTRA_ARGS=""
if [ $PROJECT_NAME -eq "taichi-nightly" ]; then
    PROJECT_TAGS="egg_info --tag-date"
fi

if [[ $OSTYPE == "linux-"* ]]; then
    EXTRA_ARGS="-p manylinux1_x86_64"
fi

python3 misc/make_changelog.py origin/master ./ True
TAICHI_CMAKE_ARGS=$CI_SETUP_CMAKE_ARGS PROJECT_NAME=$PROJECT_NAME python3 setup.py $PROJECT_TAGS bdist_wheel $EXTRA_ARGS
# Run basic cpp tests

CUR_DIR=`pwd`
TI_LIB_DIR=$CUR_DIR/python/taichi/lib ./build/taichi_cpp_tests
cat /home/dev/sccache_error
sccache -s
chmod -fR 777 $SCCACHE_DIR || true
cp dist/*.whl /wheel/
rm -f python/CHANGELOG.md
