#!/bin/bash
set -ex

export TI_SKIP_VERSION_CHECK=ON
export TI_CI=1

. $(dirname $0)/common-utils.sh


function build-and-smoke-test-android-aot-demo {
    pushd taichi
    GIT_COMMIT=$(git rev-parse HEAD | cut -c1-7)
    setup_python
    popd

    export TAICHI_REPO_DIR=$(pwd)/taichi

    git clone https://github.com/taichi-dev/taichi-aot-demo

    # Normally we checkout the master's commit Id: https://github.com/taichi-dev/taichi-aot-demo/commit/master
    # As for why we need this explicit commit Id here, refer to: https://docs.taichi-lang.org/docs/master/contributor_guide#handle-special-ci-failures
    pushd taichi-aot-demo
    git checkout bd8fb7bf30a3494f4ecc671cd4d017b926afed10
    popd

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
    git clone --reference-if-able /var/lib/git-cache https://github.com/taichi-dev/Taichi-UnityExample

    python misc/generate_unity_language_binding.py
    cp c_api/unity/*.cs Taichi-UnityExample/Assets/Taichi/Generated
    cp build/libtaichi_c_api.so Taichi-UnityExample/Assets/Plugins/Android

    export TAICHI_REPO_DIR=$(pwd)

    setup-android-ndk-env
    git clone --reference-if-able /var/lib/git-cache https://github.com/taichi-dev/taichi-unity2
    mkdir tu2-build
    pushd tu2-build
    cmake ../taichi-unity2 -DTAICHI_C_API_INSTALL_DIR=$TAICHI_REPO_DIR/_skbuild/linux-x86_64-3.9/cmake-install/c_api $ANDROID_CMAKE_ARGS
    cmake --build .
    popd
    cp tu2-build/bin/libtaichi_unity.so Taichi-UnityExample/Assets/Plugins/Android
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

$1
