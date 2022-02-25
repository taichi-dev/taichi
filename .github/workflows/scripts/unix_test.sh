#!/bin/bash
set -ex

check_in_docker() {
    # This is a temporary solution to detect in a docker, but it should work
    if [[ $(whoami) == "dev" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

export TI_SKIP_VERSION_CHECK=ON
export TI_IN_DOCKER=$(check_in_docker)

if [[ "$TI_IN_DOCKER" == "true" ]]; then
    source $HOME/miniconda/etc/profile.d/conda.sh
    conda activate "$PY"
fi
python3 -m pip install dist/*.whl
if [ -z "$GPU_TEST" ]; then
    python3 -m pip install -r requirements_test.txt
    if [[ $PY != *"3.10"* || $PY != *"py310"* ]]; then
        python3 -m pip install torch
    fi
else
    ## Only GPU machine uses system python.
    export PATH=$PATH:$HOME/.local/bin
    # pip will skip packages if already installed
    python3 -m pip install -r requirements_test.txt
fi
ti diagnose
ti changelog
echo "wanted archs: $TI_WANTED_ARCHS"

TI_PATH=$(python3 -c "import taichi;print(taichi.__path__[0])" | tail -1)
TI_LIB_DIR="$TI_PATH/_lib/runtime" ./build/taichi_cpp_tests

if [ -z "$GPU_TEST" ]; then
    python3 tests/run_tests.py -vr2 -t4 -a "$TI_WANTED_ARCHS"
else
    # only split per arch for self_hosted GPU tests
    if [[ $TI_WANTED_ARCHS == *"cuda"* ]]; then
        python3 tests/run_tests.py -vr2 -t4 -k "not torch" -a cuda
    fi
    if [[ $TI_WANTED_ARCHS == *"cpu"* ]]; then
        python3 tests/run_tests.py -vr2 -t8 -k "not torch" -a cpu
    fi
    if [[ $TI_WANTED_ARCHS == *"vulkan"* ]]; then
        python3 tests/run_tests.py -vr2 -t8 -k "not torch" -a vulkan
    fi
    if [[ $TI_WANTED_ARCHS == *"opengl"* ]]; then
        python3 tests/run_tests.py -vr2 -t4 -k "not torch" -a opengl
    fi
    # Run metal and vulkan separately so that they don't use M1 chip simultaneously.
    if [[ $TI_WANTED_ARCHS == *"metal"* ]]; then
        python3 tests/run_tests.py -vr2 -t4 -k "not torch" -a metal
    fi
    python3 tests/run_tests.py -vr2 -t1 -k "torch" -a "$TI_WANTED_ARCHS"
fi
