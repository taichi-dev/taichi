#!/bin/bash
set -ex

export TI_SKIP_VERSION_CHECK=ON
export TI_CI=1

. $(dirname $0)/common-utils.sh

cd taichi

export TAICHI_CMAKE_ARGS="$TAICHI_CMAKE_ARGS -DTI_WITH_LLVM=OFF -DTI_WITH_C_API:BOOL=ON"

setup-sccache-local
setup_python
setup-android-ndk-env

python setup.py clean
python setup.py build_ext
cd build
aarch64-linux-android-strip libtaichi_export_core.so
aarch64-linux-android-strip libtaichi_c_api.so

chmod -R 777 "$SCCACHE_DIR"
