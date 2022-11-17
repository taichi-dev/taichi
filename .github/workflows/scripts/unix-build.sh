#!/bin/bash

set -ex

. $(dirname $0)/common-utils.sh

[[ "$IN_DOCKER" == "true" ]] && cd taichi

# TODO: Move llvm installation from container image to here
if [[ "$LLVM_VERSION" == "15" ]]; then
  if [[ $OSTYPE == "linux-"* ]]; then
    wget https://github.com/ailzhang/torchhub_example/releases/download/0.2/taichi-llvm-15-linux.zip
    unzip taichi-llvm-15-linux.zip && rm taichi-llvm-15-linux.zip
    export PATH="$PWD/taichi-llvm-15/bin:$PATH"
  elif [ "$(uname -s):$(uname -m)" == "Darwin:arm64" ]; then
    # The following commands are done manually to save time.
    #wget https://github.com/ailzhang/torchhub_example/releases/download/0.2/taichi-llvm-15-m1.zip
    #unzip taichi-llvm-15-m1.zip && rm taichi-llvm-15-m1.zip
    export PATH="/Users/github/taichi-llvm-15-m1/bin:$PATH"
  elif [ "$(uname -s):$(uname -m)" == "Darwin:x86_64" ]; then
    # The following commands are done manually to save time.
    #wget https://github.com/ailzhang/torchhub_example/releases/download/0.2/llvm-15-macos10.15.zip
    #unzip llvm-15-macos10.15.zip && rm llvm-15-macos10.15.zip
    export LLVM_DIR="~/llvm-15-mac10.15/"
  fi
  export TAICHI_CMAKE_ARGS="$TAICHI_CMAKE_ARGS -DTI_LLVM_15:BOOL=ON"
fi

build_taichi_wheel() {
    python3 -m pip install -r requirements_dev.txt
    git fetch origin master --tags
    PROJECT_TAGS=""
    EXTRA_ARGS=""
    if [ "$PROJECT_NAME" = "taichi-nightly" ]; then
        PROJECT_TAGS="egg_info --tag-date"
        # Include C-API in nightly builds
        TAICHI_CMAKE_ARGS="$TAICHI_CMAKE_ARGS -DTI_WITH_C_API:BOOL=ON"
    fi

    if [[ $OSTYPE == "linux-"* ]]; then
        if [ -f /etc/centos-release ] ; then
            EXTRA_ARGS="-p manylinux2014_x86_64"
        else
            EXTRA_ARGS="-p manylinux_2_27_x86_64"
        fi
    fi
    python3 misc/make_changelog.py --ver origin/master --repo_dir ./ --save

    python3 setup.py $PROJECT_TAGS bdist_wheel $EXTRA_ARGS
    sccache -s || true
}

fix-build-cache-permission

setup-sccache-local
setup_python

build_taichi_wheel
NUM_WHL=$(ls dist/*.whl | wc -l)
if [ $NUM_WHL -ne 1 ]; then echo "ERROR: created more than 1 whl." && exit 1; fi

chmod -R 777 "$SCCACHE_DIR"
rm -f python/CHANGELOG.md
