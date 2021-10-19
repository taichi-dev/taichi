#!/bin/bash

SHA=$1
PY=$2

set -ex
source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY 

# Build Taichi from source
git clone --recursive https://github.com/taichi-dev/taichi --branch=master
cd taichi
git checkout $SHA
python3 -m pip install --user -r requirements_dev.txt -i http://repo.taichigraphics.com/repository/pypi/simple --trusted-host repo.taichigraphics.com 
# Update Torch version, otherwise cuda tests fail. See #2969.
python3 -m pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html -i http://repo.taichigraphics.com/repository/pypi/simple --trusted-host repo.taichigraphics.com
TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON" python3 setup.py develop --user

# Link Taichi source repo to Python Path
export PATH="/home/dev/taichi/bin:$PATH"
export TAICHI_REPO_DIR="/home/dev/taichi/"
export PYTHONPATH="$TAICHI_REPO_DIR/python:$PYTHONPATH"

# Add Docker specific ENV
export TI_IN_DOCKER=true

# Run tests
ti diagnose
ti test -vr2 -t2 -k "not ndarray and not torch"
ti test -vr2 -t1 -k "ndarray or torch"
