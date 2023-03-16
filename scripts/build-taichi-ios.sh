#!/bin/bash
set -e
#CLANG_EXECUTABLE="$(brew list llvm@15 | grep clang++ | head -1)"

if [[ -z "${TAICHI_REPO_DIR}" ]]; then
    echo "Please set TAICHI_REPO_DIR env variable"
    exit
else
    echo "TAICHI_REPO_DIR is set to ${TAICHI_REPO_DIR}"
fi

if [[ ! -f "tmp/ios.toolchain.cmake" ]]; then
    if [[ ! -e "tmp" ]]; then
        mkdir tmp
    fi
    curl https://raw.githubusercontent.com/leetal/ios-cmake/master/ios.toolchain.cmake -o tmp/ios.toolchain.cmake
fi

rm -rf build-taichi-ios-arm64
mkdir build-taichi-ios-arm64
pushd build-taichi-ios-arm64
# See https://stackoverflow.com/questions/12630970/compiling-for-ios-with-cmake
cmake $TAICHI_REPO_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="../tmp/ios.toolchain.cmake" \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED="NO" \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED="NO" \
    -DENABLE_BITCODE=ON \
    -DENABLE_ARC=OFF \
    -DDEPLOYMENT_TARGET=13.0 \
    -DPLATFORM=OS64 \
    -DCMAKE_INSTALL_PREFIX="./install" \
    -DCLANG_EXECUTABLE=$CLANG_EXECUTABLE \
    -G "Xcode" \
    -DUSE_STDCPP=ON \
    -DTI_WITH_C_API=ON \
    -DTI_WITH_METAL=ON \
    -DTI_WITH_VULKAN=OFF \
    -DTI_WITH_OPENGL=OFF \
    -DTI_WITH_LLVM=OFF \
    -DTI_WITH_CUDA=OFF \
    -DTI_WITH_PYTHON=OFF \
    -DTI_WITH_GGUI=OFF \
    -DTI_WITH_CC=OFF
cmake --build . -t taichi_c_api
cmake --build . -t install
popd

python $PWD/scripts/archive-minimal-static.py
