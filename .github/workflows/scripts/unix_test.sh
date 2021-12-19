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
else
    ## Only GPU machine uses system python.
    export PATH=$PATH:$HOME/.local/bin
fi
ti diagnose
ti changelog
echo "wanted archs: $TI_WANTED_ARCHS"

TI_PATH=$(python3 -c "import taichi;print(taichi.__path__[0])" | tail -1)
TI_LIB_DIR="$TI_PATH/_lib/runtime" ./build/taichi_cpp_tests

if [ -z "$GPU_TEST" ]; then
    python3 tests/run_tests.py -vr2 -t2 -a "$TI_WANTED_ARCHS"
else
    python3 tests/run_tests.py -vr2 -t2 -k "not ndarray and not torch" -a "$TI_WANTED_ARCHS"
    python3 tests/run_tests.py -vr2 -t1 -k "ndarray or torch" -a "$TI_WANTED_ARCHS"
fi
