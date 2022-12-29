#!/bin/bash

# TODO: move inside docker
wget https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip
unzip taichi-llvm-15-linux.zip && rm taichi-llvm-15-linux.zip
export LLVM_DIR="$PWD/taichi-llvm-15"

pushd taichi
python3 -m pip install -r requirements_dev.txt

CI_SETUP_CMAKE_ARGS=$1
export CI_SETUP_CMAKE_ARGS
python3 ./scripts/run_clang_tidy.py $PWD/taichi -clang-tidy-binary clang-tidy-10 -header-filter=$PWD/taichi -j2
popd
