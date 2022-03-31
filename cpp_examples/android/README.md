Configuration
=============

In order to build and run the Android C++ Example, it is requires to build the exported Taichi library for Android (see Build section) and update the following file with the proper path *if needed*: `app/src/main/cpp/CMakeLists.txt`  
By default, the library is set to use the `build` directory from the root directory of Taichi.


Build
=====

This C++ example requires a specific version of Taichi cross-compiled for Android, this can be achieved with the following command:

    export ANDROID_SDK_ROOT=$HOME/Android/Sdk/ // Update this line according to your system
    TAICHI_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_SDK_ROOT}/ndk/22.1.7171670/build/cmake/android.toolchain.cmake -DANDROID_NATIVE_API_LEVEL=29 -DANDROID_ABI=arm64-v8a -DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=OFF -DTI_EXPORT_CORE:BOOL=ON" python3 setup.py build_ext

The next steps is to make sure to compile the Python AOT module located here:

    android/app/src/main/assets/mpm88
    
This needs to be compiled with the Desktop version of Taichi by running the command:

    python3 mpm88_aot.py

After that the following content should be created:

    init_c58_0_k0000_vk_0_t00.spv
    metadata.json
    metadata.tcb
    mpm88_aot.py
    substep_c56_0_k0004_vk_0_t00.spv
    substep_c56_0_k0004_vk_1_t01.spv
    substep_c56_0_k0004_vk_2_t02.spv
    substep_c56_0_k0004_vk_3_t03.spv

Then the Android application can be compiled:

    ./gradlew assembleDebug

Install
=======

Installing the APK is a straightforward process and can be achieved running the following command:

    adb install ./app/build/outputs/apk/debug/app-debug.apk

Once the application is running, some of the logs related to the application can be monitored with logcat:

    adb logcat -s taichi vulkan native-activity ALooper VALIDATION AndroidRuntime "DEBUG" "*:F"
