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

    rm -rf taichi-aot-demo
    # IF YOU PIN THIS TO A COMMIT/BRANCH, YOU'RE RESPONSIBLE TO REVERT IT BACK TO MASTER ONCE MERGED.
    git clone https://github.com/taichi-dev/taichi-aot-demo

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
    git clone --reference-if-able /var/lib/git-cache -b upgrade-modules2 https://github.com/taichi-dev/Taichi-UnityExample

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
    pushd taichi
    setup_python
    popd

    cd taichi
    pushd Taichi-UnityExample
    python scripts/implicit_fem.cgraph.py --aot
    popd
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
    git clone --recursive --depth=1 https://github.com/taichi-dev/taichi-aot-demo
    cd taichi-aot-demo
    mkdir build
    pushd build
    export TAICHI_C_API_INSTALL_DIR=$(find $TAICHI_REPO_DIR -name cmake-install -type d | head -n 1)/c_api
    cmake $ANDROID_CMAKE_ARGS ..
    make -j
    export PATH=/android-sdk/platform-tools:$PATH
    grab-android-bot
    trap release-android-bot EXIT
    adb connect $BOT

    # clear temporary test folder
    adb shell "rm -rf /data/local/tmp/*"

    cd headless
    BINARIES=$(ls E*)
    for b in $BINARIES; do
        adb push $b /data/local/tmp
    done
    adb push $TAICHI_C_API_INSTALL_DIR/lib/libtaichi_c_api.so /data/local/tmp

    popd # build

    for dir in ?_*; do
        adb push $dir /data/local/tmp
    done

    for b in $BINARIES; do
        adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=\$(pwd) ./$b"
        adb pull /data/local/tmp/0001.bmp $b.bmp
    done

    for b in $BINARIES; do
        if [[ $(cmp -l $b.bmp ci/headless-truths/$b.bmp | wc -l) -gt 300 ]]; then
            echo "Above threshold: $b"
            exit 1
        fi
    done
}

$1
