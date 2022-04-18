#!/bin/bash
set -ex

export TI_SKIP_VERSION_CHECK=ON
export TI_CI=1

if [ -f "/home/dev/miniconda/etc/profile.d/conda.sh" ] && [ -n "$PY" ] ; then
    source "/home/dev/miniconda/etc/profile.d/conda.sh"
    conda activate "$PY"
fi
python3 -m pip install dist/*.whl
if [ -z "$GPU_TEST" ]; then
    python3 -m pip install -r requirements_test.txt
    python3 -m pip install "torch; python_version < '3.10'"
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
    if [[ $PLATFORM == *"m1"* ]]; then
	# Split per arch to avoid flaky test
        python3 tests/run_tests.py -vr2 -t4 -k "not torch" -a cpu
        # Run metal and vulkan separately so that they don't use M1 chip simultaneously.
        python3 tests/run_tests.py -vr2 -t4 -k "not torch" -a vulkan
        python3 tests/run_tests.py -vr2 -t2 -k "not torch" -a metal
        python3 tests/run_tests.py -vr2 -t1 -k "torch" -a "$TI_WANTED_ARCHS"
    else
        python3 tests/run_tests.py -vr2 -t4 -a "$TI_WANTED_ARCHS"
    fi
else
    # Split per arch to increase parallelism for linux GPU tests
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
    python3 tests/run_tests.py -vr2 -t1 -k "torch" -a "$TI_WANTED_ARCHS"
fi
