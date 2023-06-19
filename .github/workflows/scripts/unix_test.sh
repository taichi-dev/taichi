#!/bin/bash
set -ex

. $(dirname $0)/common-utils.sh

export PYTHONUNBUFFERED=1

export TAICHI_AOT_FOLDER_PATH="taichi/tests"
export TI_SKIP_VERSION_CHECK=ON
export LD_LIBRARY_PATH=$PWD/build/:$LD_LIBRARY_PATH
export TI_OFFLINE_CACHE_FILE_PATH=$PWD/.cache/taichi


# Disable compat tests to save time.
# According to @PENGUINLIONG, this is currently not doing its job, and
# will be refactored.
# pip3 install -i https://pypi.taichi.graphics/simple/ taichi-nightly
[[ "$IN_DOCKER" == "true" ]] && cd taichi
# python3 tests/generate_compat_test_modules.py
# python3 -m pip uninstall taichi-nightly -y

python3 .github/workflows/scripts/build.py --permissive --write-env=/tmp/ti-env.sh
. /tmp/ti-env.sh

install_taichi_wheel

# python3 tests/run_c_api_compat_test.py

ti diagnose
ti changelog
echo "wanted archs: $TI_WANTED_ARCHS"


if [ -z "$TI_SKIP_CPP_TESTS" ]; then
    echo "Running cpp tests on platform:" "${PLATFORM}"
    python3 tests/run_tests.py --cpp -vr2 -t4 ${EXTRA_TEST_MARKERS:+-m "$EXTRA_TEST_MARKERS"}
fi


if [ "$TI_RUN_RELEASE_TESTS" == "1" ]; then
    python3 -m pip install PyYAML
    git clone https://github.com/taichi-dev/taichi-release-tests
    pushd taichi-release-tests
    git checkout 20230619
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
    git clone --depth=1 https://github.com/taichi-dev/games201
    git clone --depth=1 https://github.com/taichiCourse01/--Galaxy
    git clone --depth=1 https://github.com/taichiCourse01/--Shadertoys
    git clone --depth=1 https://github.com/taichiCourse01/taichi_ray_tracing
    popd

    pushd repos/difftaichi
    pip install -r requirements.txt
    popd

    python3 run.py --log=DEBUG --runners 1 timelines
    popd
fi

function run-it {
    ARCH=$1
    PARALLELISM=$2
    KEYS=${3:-"not torch and not paddle"}

    if [[ $TI_WANTED_ARCHS == *"$1"* ]]; then
        python3 tests/run_tests.py -vr2 -t$PARALLELISM -k "$KEYS" -m "not run_in_serial ${EXTRA_TEST_MARKERS:+and $EXTRA_TEST_MARKERS}" -a $ARCH
        python3 tests/run_tests.py -vr2 -t1 -k "$KEYS" -m "run_in_serial ${EXTRA_TEST_MARKERS:+and $EXTRA_TEST_MARKERS}" -a $ARCH
    fi
}

# Workaround for 'cannot allocate memory in static TLS block' issue
# During the test, the library below is loaded and unloaded multiple times,
# each time leaking some static TLS memory, and eventually depleting it.
# Preloading the library here to avoid unloading it during the test.
LIBNVIDIA_TLS=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-tls.so.* 2>/dev/null || true)
if [ ! -z $LIBNVIDIA_TLS ]; then
    export LD_PRELOAD=$LIBNVIDIA_TLS${LD_PRELOAD:+:$LD_PRELOAD}
fi


N=$(nproc)

# FIXME: This variable (GPU_TEST) only adds confusion, should refactor it out.
if [ -z "$GPU_TEST" ]; then
    if [[ $PLATFORM == *"m1"* ]]; then
        run-it cpu    4
        run-it vulkan 4
        run-it metal  2

        run-it cpu    1 "torch"
        run-it vulkan 1 "torch"
        run-it metal  1 "torch"
    else
        echo "::warning:: Hitting Running CPU tests only"
        # Fail fast, give priority to the error-prone tests
        if [[ $OSTYPE == "linux-"* ]]; then
            run-it cpu 1 "paddle"
        fi
        run-it cpu $N
        run-it cpu 1 "torch"
    fi
else
    run-it cpu    $N
    run-it cuda   8
    run-it vulkan 8
    run-it opengl 4
    run-it gles   4
    run-it amdgpu 8

    run-it cpu    1 "torch"
    run-it cuda   1 "torch"
    run-it vulkan 1 "torch"
    run-it opengl 1 "torch"
    run-it gles   1 "torch"
    # run-it amdgpu 1 "torch"

    # Paddle's paddle.fluid.core.Tensor._ptr() is only available on develop branch, and CUDA version on linux will get error `Illegal Instruction`
fi
