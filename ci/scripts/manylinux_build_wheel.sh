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
# Add Docker specific ENV
export TI_IN_DOCKER=true

# TODO, unify this step with wheel build, check #3537
TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=OFF -DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_CC:BOOL=OFF" python3 setup.py install
# build.py is to be removed
#cd python && python build.py build
