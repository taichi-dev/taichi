#!/bin/bash
set -ex

. $(dirname $0)/common-utils.sh

export PYTHONUNBUFFERED=1

export TAICHI_AOT_FOLDER_PATH="taichi/tests"
export TI_SKIP_VERSION_CHECK=ON
export LD_LIBRARY_PATH=$PWD/build/:$LD_LIBRARY_PATH
export TI_OFFLINE_CACHE_FILE_PATH=$PWD/.cache/taichi


pip3 install -i https://pypi.taichi.graphics/simple/ taichi-nightly
[[ "$IN_DOCKER" == "true" ]] && cd taichi
python3 tests/generate_compat_test_modules.py
python3 -m pip uninstall taichi-nightly -y

setup_python

install_taichi_wheel

python3 tests/run_c_api_compat_test.py

ti diagnose
ti changelog
echo "wanted archs: $TI_WANTED_ARCHS"

if [ "$TI_RUN_RELEASE_TESTS" == "1" ]; then
    python3 -m pip install PyYAML
    git clone https://github.com/taichi-dev/taichi-release-tests
    pushd taichi-release-tests
    git checkout 20230130
    mkdir -p repos/taichi/python/taichi
    EXAMPLES=$(cat <<EOF | python3 | tail -n 1
import taichi.examples
print(taichi.examples.__path__[0])
EOF
)
    ln -sf $EXAMPLES repos/taichi/python/taichi/examples
    pushd repos
    git clone --depth=1 https://github.com/taichi-dev/quantaichi
    git clone --depth=1 https://github.com/taichi-dev/difftaichi
    popd

    pushd repos/difftaichi
    pip install -r requirements.txt
    popd

    python3 run.py --log=DEBUG --runners 1 timelines
    popd
fi

if [ -z "$TI_SKIP_CPP_TESTS" ]; then
    echo "Running cpp tests on platform:" "${PLATFORM}"
    # Temporary hack before CI Pipeline Overhaul
    if [[ $PLATFORM == *"linux"* ]]; then
        if nvidia-smi -L | grep "Tesla P4"; then
            python3 tests/run_tests.py --cpp -vr2 -t6 -m "not sm70"
        else
            python3 tests/run_tests.py --cpp -vr2 -t6
        fi
    else
        python3 tests/run_tests.py --cpp -vr2 -t6
    fi
fi

function run-it {
    ARCH=$1
    PARALLELISM=$2
    KEYS=${3:-"not torch and not paddle"}

    if [[ $TI_WANTED_ARCHS == *"$1"* ]]; then
        python3 tests/run_tests.py -vr2 -t$PARALLELISM -k "$KEYS" -m "not run_in_serial" -a $ARCH
        python3 tests/run_tests.py -vr2 -t1 -k "$KEYS" -m "run_in_serial" -a $ARCH
    fi
}

if [ -z "$GPU_TEST" ]; then
    if [[ $PLATFORM == *"m1"* ]]; then
        run-it cpu 4
        run-it vulkan 4
        run-it metal 2

        python3 tests/run_tests.py -vr2 -t1 -k "torch" -a "$TI_WANTED_ARCHS"
    else
        # Fail fast, give priority to the error-prone tests
        if [[ $OSTYPE == "linux-"* ]]; then
            python3 tests/run_tests.py -vr2 -t1 -k "paddle" -a "$TI_WANTED_ARCHS"
        fi
        python3 tests/run_tests.py -vr2 -t4 -k "not paddle" -a "$TI_WANTED_ARCHS"
    fi
elif [ ! -z "$AMDGPU_TEST" ]; then
    run-it cpu    $(nproc)
    run-it amdgpu 8
else
    run-it cuda   8
    run-it cpu    $(nproc)
    run-it vulkan 8
    run-it opengl 4
    run-it gles   4

    python3 tests/run_tests.py -vr2 -t1 -k "torch" -a "$TI_WANTED_ARCHS"
    # Paddle's paddle.fluid.core.Tensor._ptr() is only available on develop branch, and CUDA version on linux will get error `Illegal Instruction`

    # FIXME: Running gles test separatelyfor now, add gles to TI_WANTED_ARCHS once running "-a vulkan,opengl,gles" is fixed
    if [[ $TI_WANTED_ARCHS == *opengl* ]]; then
      python3 tests/run_tests.py -vr2 -t1 -k "torch" -a gles
    fi
fi
