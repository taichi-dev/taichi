#!/bin/bash

CI_SETUP_CMAKE_ARGS=$1

cd taichi
python3 -m pip install -r requirements_dev.txt

mkdir build && cd build
cmake $CI_SETUP_CMAKE_ARGS ..

cd ..
python3 ./scripts/run_clang_tidy.py $PWD/taichi -clang-tidy-binary clang-tidy-10 -checks=-*,performance-inefficient-string-concatenation -header-filter=$PWD/taichi -p $PWD/build -j2
