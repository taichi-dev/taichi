#!/bin/bash
set -ex

export TI_SKIP_VERSION_CHECK=ON
export TI_CI=1

export TAICHI_AOT_DEMO_URL=https://github.com/taichi-dev/taichi-aot-demo
export TAICHI_AOT_DEMO_BRANCH=master

export TAICHI_UNITY2_URL=https://github.com/taichi-dev/taichi-unity2
export TAICHI_UNITY2_BRANCH=main

export TAICHI_UNITY_EXAMPLE_URL=https://github.com/damnkk/Taichi-UnityExample.git
export TAICHI_UNITY_EXAMPLE_BRANCH=mpm88_fix1

. $(dirname $0)/common-utils.sh


function build-and-smoke-test-android-aot-demo {
    pushd taichi
    GIT_COMMIT=$(git rev-parse HEAD | cut -c1-7)
    setup_python
    popd

    export TAICHI_REPO_DIR=$(pwd)/taichi

    rm -rf taichi-aot-demo
    # IF YOU PIN THIS TO A COMMIT/BRANCH, YOU'RE RESPONSIBLE TO REVERT IT BACK TO MASTER ONCE MERGED.
    git clone --depth=1 -b "$TAICHI_AOT_DEMO_BRANCH" "$TAICHI_AOT_DEMO_URL"

    APP_ROOT=taichi-aot-demo/implicit_fem
    ANDROID_APP_ROOT=$APP_ROOT/android
    JNI_PATH=$ANDROID_APP_ROOT/app/src/main/jniLibs/arm64-v8a/

    pip install /taichi-wheel/*.whl
    pushd $APP_ROOT/python
    sudo chmod 0777 $HOME/.cache
    python implicit_fem.py --aot
    popd
    mkdir -p $JNI_PATH
    cp taichi/build/libtaichi_export_core.so $JNI_PATH
    cd $ANDROID_APP_ROOT
    sed -i "s/TaichiAOT/AOT-$GIT_COMMIT/g" app/src/main/res/values/strings.xml
    ./gradlew build

    run-android-app \
        app/build/outputs/apk/debug/app-debug.apk \
        com.taichigraphics.aot_demos.implicit_fem/android.app.NativeActivity
}

function prepare-unity-build-env {
    cd taichi

    # Dependencies
    git clone --reference-if-able /var/lib/git-cache -b "$TAICHI_UNITY_EXAMPLE_BRANCH" "$TAICHI_UNITY_EXAMPLE_URL"

    python misc/generate_unity_language_binding.py
    cp c_api/unity/*.cs Taichi-UnityExample/Assets/Taichi/Generated
    cp build/libtaichi_c_api.so Taichi-UnityExample/Assets/Plugins/Android

    export TAICHI_REPO_DIR=$(pwd)

    setup-android-ndk-env
    git clone --reference-if-able /var/lib/git-cache -b "$TAICHI_UNITY2_BRANCH" "$TAICHI_UNITY2_URL"
    mkdir tu2-build
    pushd tu2-build
    cmake ../taichi-unity2 -DTAICHI_C_API_INSTALL_DIR=$TAICHI_REPO_DIR/_skbuild/linux-x86_64-3.9/cmake-install/c_api $ANDROID_CMAKE_ARGS
    cmake --build .
    popd
    cp tu2-build/bin/libtaichi_unity.so Taichi-UnityExample/Assets/Plugins/Android

    pushd Taichi-UnityExample
    pip install /taichi-wheel/*.whl
    python scripts/implicit_fem.cgraph.py --aot
    popd
}

function build-unity-demo {
    cd taichi
    mkdir -p Taichi-UnityExample/Assets/Editor
    cp -a /UnityBuilderAction Taichi-UnityExample/Assets/Editor/
    unity-editor \
        -logfile /dev/stdout \
        -quit \
        -customBuildName Android \
        -projectPath Taichi-UnityExample \
        -buildTarget Android \
        -customBuildTarget Android \
        -customBuildPath build/Android/Android.apk \
        -executeMethod UnityBuilderAction.Builder.BuildProject \
        -buildVersion 1.0.0-ci \
        -androidVersionCode 1000000 \
        -androidKeystoreName ~/.android/debug.keystore \
        -androidKeystorePass android \
        -androidKeyaliasName androiddebugkey \
        -androidKeyaliasPass android
}

function smoke-test-unity-demo {
    run-android-app \
        taichi/Taichi-UnityExample/build/Android/Android.apk \
        com.TaichiGraphics.TaichiUnityExample/com.unity3d.player.UnityPlayerActivity \
        6
}

function build-and-test-headless-demo {
    setup-android-ndk-env

    pushd taichi
    setup_python
    popd

    export TAICHI_REPO_DIR=$(pwd)/taichi

    pushd taichi
    pip install /taichi-wheel/*.whl
    sudo chmod 0777 $HOME/.cache
    popd

    rm -rf taichi-aot-demo
    git clone --recursive --depth=1 -b "$TAICHI_AOT_DEMO_BRANCH" "$TAICHI_AOT_DEMO_URL"
    cd taichi-aot-demo

    . $(pwd)/ci/test_utils.sh

    # Build demos
    export TAICHI_C_API_INSTALL_DIR=$(find $TAICHI_REPO_DIR -name cmake-install -type d | head -n 1)/c_api
    build_demos "$ANDROID_CMAKE_ARGS"

    export PATH=/android-sdk/platform-tools:$PATH
    grab-android-bot
    trap release-android-bot EXIT
    adb connect $BOT

    # Clear temporary test folder
    adb shell "rm -rf /data/local/tmp/* && mkdir /data/local/tmp/build"

    # Push all binaries and shaders to the phone
    pushd ci
    adb push ./*.sh /data/local/tmp
    popd
    adb push $TAICHI_C_API_INSTALL_DIR/lib/libtaichi_c_api.so /data/local/tmp
    adb push ./build/headless /data/local/tmp/build
    adb push ./build/tutorial /data/local/tmp/build
    for dir in ?_*; do
        adb push $dir /data/local/tmp
    done

    # Run demos
    adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=\$(pwd) ./run_demos.sh"

    # Pull output images and compare with groundtruth
    rm -rf output
    mkdir output

    adb pull /data/local/tmp/output .
    compare_to_groundtruth android
}

function build-and-test-headless-demo-desktop {
    pushd taichi
    setup_python
    python3 -m pip install dist/*.whl
    sudo chmod 0777 $HOME/.cache
    export TAICHI_REPO_DIR=$(pwd)
    popd

    rm -rf taichi-aot-demo
    git clone --recursive --depth=1 -b "$TAICHI_AOT_DEMO_BRANCH" "$TAICHI_AOT_DEMO_URL"
    cd taichi-aot-demo

    TAICHI_C_API_INSTALL_DIR=$(find $TAICHI_REPO_DIR -name cmake-install -type d | head -n 1)/c_api
    python3 -m pip install -r ci/requirements.txt
    python3 ci/run_tests.py -l $TAICHI_C_API_INSTALL_DIR
}

function check-c-api-export-symbols {
    cd taichi
    TAICHI_REPO_DIR=$(pwd)
    TAICHI_C_API_DIR=$(find $TAICHI_REPO_DIR -name libtaichi_c_api.* | head -n 1)

    # T: global functions
    # B: global variables (uninitialized)
    # D: global variables (initialized)
    EXPORT_SYM=" T \| B \| D "

    # Note: this has to be consistent with the version scripts (export_symbol_linux.ld, export_symbol_mac.ld)
    CAPI_SYM=" _\?ti_"
    CAPI_UTILS_SYM=" capi::utils::"

    NUM_LEAK_SYM=$(nm -C --extern-only ${TAICHI_C_API_DIR} | grep "${EXPORT_SYM}" | grep -v "${CAPI_SYM}" | grep -v "${CAPI_UTILS_SYM}" | wc -l)
    if [ ${NUM_LEAK_SYM} -gt 0 ]; then
        echo "Following symbols leaked from libtaichi_c_api: "
        nm -C --extern-only ${TAICHI_C_API_DIR} | grep "${EXPORT_SYM}" | grep -v "${CAPI_SYM}" | grep -v "${CAPI_UTILS_SYM}"
        exit 1
    fi
}

$1
