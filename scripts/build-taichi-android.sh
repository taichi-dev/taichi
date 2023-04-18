#!/bin/bash
set -e

CLANG_EXECUTABLE=$(find $ANDROID_NDK_ROOT -name "clang++")

if [[ -z "${TAICHI_REPO_DIR}" ]]; then
    echo "Please set TAICHI_REPO_DIR env variable"
    exit
else
    echo "TAICHI_REPO_DIR is set to ${TAICHI_REPO_DIR}"
fi

if [[ -z "${ANDROID_NDK_ROOT}" ]]; then
    echo "Please set ANDROID_NDK_ROOT env variable"
    exit
else
    echo "ANDROID_NDK_ROOT is set to ${ANDROID_NDK_ROOT}"
fi

#rm -rf build-taichi-android-aarch64
#mkdir build-taichi-android-aarch64
pushd build-taichi-android-aarch64
cmake $TAICHI_REPO_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="./install" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DCLANG_EXECUTABLE=$CLANG_EXECUTABLE \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-26 \
    -G "Ninja" \
    -DTI_WITH_CUDA=OFF \
    -DTI_WITH_CUDA_TOOLKIT=OFF \
    -DTI_WITH_C_API=ON \
    -DTI_WITH_DX11=OFF \
    -DTI_WITH_LLVM=OFF \
    -DTI_WITH_METAL=OFF \
    -DTI_WITH_OPENGL=OFF \
    -DTI_WITH_PYTHON=OFF \
    -DTI_WITH_VULKAN=ON

cmake --build . -t taichi_c_api
cmake --build . -t install
popd
