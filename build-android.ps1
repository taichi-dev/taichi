if (-not(Test-Path "build-android-aarch64")) {
    New-Item "build-android-aarch64" -ItemType Directory
}

Push-Location "build-android-aarch64"
cmake -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="${pwd}/../build-android-aarch64/install" `
    -DCMAKE_TOOLCHAIN_FILE="$env:ANDROID_NDK/build/cmake/android.toolchain.cmake" `
    -DCLANG_EXECUTABLE="${pwd}/../tmp/taichi-clang/bin/clang++.exe" `
    -DANDROID_ABI="arm64-v8a" `
    -DANDROID_PLATFORM=android-26 `
    -G "Ninja" `
    -DTI_WITH_CC=OFF `
    -DTI_WITH_CUDA=OFF `
    -DTI_WITH_CUDA_TOOLKIT=OFF `
    -DTI_WITH_C_API=ON `
    -DTI_WITH_DX11=OFF `
    -DTI_WITH_LLVM=OFF `
    -DTI_WITH_METAL=OFF `
    -DTI_WITH_OPENGL=OFF `
    -DTI_WITH_PYTHON=OFF `
    -DTI_WITH_VULKAN=ON `
    ..
cmake --build . -t taichi_c_api
cmake --build . -t install
Pop-Location
