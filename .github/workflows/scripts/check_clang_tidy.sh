#!/bin/bash

# we need to generate a special file "compile_commands.json"
# to do this, we need to run cmake with the following options:
# -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# I'm fairly sure that we don't need to run the actual build,
# but it's not obvious to me how to do this. So, I'm just going 
# to run a full build for now, and we can FIXME this later.

set -ex

sudo apt-get update
sudo apt-get install -y clang-tidy-20
git submodule update --init --recursive

sudo apt install -y \
    freeglut3-dev \
    libglfw3-dev \
    libglm-dev \
    libglu1-mesa-dev \
    libwayland-dev \
    libx11-xcb-dev \
    libxcb-dri3-dev \
    libxcb-ewmh-dev \
    libxcb-keysyms1-dev \
    libxcb-randr0-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    pybind11-dev \
    libc++-15-dev \
    libc++abi-15-dev \
    clang-15 \
    libclang-common-15-dev \
    libclang-cpp15 \
    libclang1-15 \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip

pip install scikit-build
export TAICHI_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
./build.py wheel
python ./scripts/run_clang_tidy.py $PWD/taichi -clang-tidy-binary clang-tidy-14 -header-filter=$PWD/taichi -j2
