#!/bin/bash

set -ex

# Parse ARGs
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            SHA)              SHA=${VALUE} ;;
            PY)               PY=${VALUE} ;;
            *)
    esac
done

source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

# Build Taichi from source
git clone --recursive https://github.com/taichi-dev/taichi --branch=master
cd taichi
git checkout $SHA
python3 -m pip install -r requirements_dev.txt -i http://repo.taichigraphics.com/repository/pypi/simple --trusted-host repo.taichigraphics.com
# Update Torch version, otherwise cuda tests fail. See #2969.
python3 -m pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html -i http://repo.taichigraphics.com/repository/pypi/simple --trusted-host repo.taichigraphics.com
TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON" python3 setup.py install

# Add Docker specific ENV
export TI_IN_DOCKER=true

# Run tests
ti diagnose
# Paddle's paddle.fluid.core.Tensor._ptr() is only available on develop branch, and CUDA version on linux will get error `Illegal Instruction`
python tests/run_tests.py -vr2 -t2 -k "not ndarray and not torch and not paddle"
python tests/run_tests.py -vr2 -t1 -k "ndarray or torch"
