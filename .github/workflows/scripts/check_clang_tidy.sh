#!/bin/bash

apt install -y clang-tidy-10 libxrandr-dev

cd taichi
python3 -m pip install -r requirements_dev.txt

mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

cd ..
python3 ./scripts/run-clang-tidy.py $PWD/taichi -checks=-*,performance-inefficient-string-concatenation -header-filter=$PWD/taichi -p $PWD/build -j2
