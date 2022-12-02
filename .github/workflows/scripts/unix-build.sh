#!/bin/bash

set -ex

. $(dirname $0)/common-utils.sh

[[ "$IN_DOCKER" == "true" ]] && cd taichi

if [[ $OSTYPE == "linux-"* ]]; then
  if [ -z "$AMDGPU_TEST"]; then
    if [ ! -d ~/taichi-llvm-15 ]; then
      pushd ~
      if [ -f /etc/centos-release ] ; then
        # FIXIME: prebuilt llvm15 on ubuntu didn't work on manylinux image of centos. Once that's fixed, remove this hack.
        wget https://github.com/ailzhang/torchhub_example/releases/download/0.3/taichi-llvm-15-linux.zip
      else
        wget https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip
      fi
      unzip taichi-llvm-15-linux.zip && rm taichi-llvm-15-linux.zip
      popd
    fi
    export LLVM_DIR="$HOME/taichi-llvm-15"
  else
    sudo ln -s /usr/bin/clang++-10 /usr/bin/clang++
    sudo ln -s /usr/bin/clang-10 /usr/bin/clang
    sudo ln -s /usr/bin/ld.lld-10 /usr/bin/ld.lld
    export LLVM_DIR="/taichi-llvm-15.0.0-linux"
  fi

    
elif [ "$(uname -s):$(uname -m)" == "Darwin:arm64" ]; then
  # The following commands are done manually to save time.
  if [ ! -d ~/taichi-llvm-15-m1 ]; then
    pushd ~
    wget https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-m1.zip
    unzip taichi-llvm-15-m1.zip && rm taichi-llvm-15-m1.zip
    popd
  fi

  export LLVM_DIR="$HOME/taichi-llvm-15-m1"
elif [ "$(uname -s):$(uname -m)" == "Darwin:x86_64" ]; then
  # The following commands are done manually to save time.
  if [ ! -d ~/llvm-15-mac10.15 ]; then
    pushd ~
    wget https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/llvm-15-mac10.15.zip
    unzip llvm-15-mac10.15.zip && rm llvm-15-mac10.15.zip
    popd
  fi

  export LLVM_DIR="$HOME/llvm-15-mac10.15/"
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

    if ["$AMDGPU_TEST"]; then
      TAICHI_CMAKE_ARGS="$TAICHI_CMAKE_ARGS -DTI_WITH_CUDA:BOOL=OFF"
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
