#!/bin/bash
set -ex

export TI_SKIP_VERSION_CHECK=ON
export TI_CI=1

# IF YOU PIN THIS TO A COMMIT/BRANCH, YOU'RE RESPONSIBLE TO REVERT IT BACK TO MASTER ONCE MERGED.
export TAICHI_AOT_DEMO_URL=https://github.com/taichi-dev/taichi-aot-demo
export TAICHI_AOT_DEMO_BRANCH=master

export TAICHI_UNITY2_URL=https://github.com/taichi-dev/taichi-unity2
export TAICHI_UNITY2_BRANCH=main

export TAICHI_UNITY_EXAMPLE_URL=https://github.com/taichi-dev/Taichi-UnityExample
export TAICHI_UNITY_EXAMPLE_BRANCH=main

. $(dirname $0)/common-utils.sh


function build-and-smoke-test-android-aot-demo {
    pushd taichi
    GIT_COMMIT=$(git rev-parse HEAD | cut -c1-7)
    python3 .github/workflows/scripts/build.py android --permissive --write-env=/tmp/ti-aot-env.sh
    . /tmp/ti-aot-env.sh
    popd

    export TAICHI_REPO_DIR=$(pwd)/taichi

    rm -rf taichi-aot-demo
    git clone --recursive --jobs=4 --depth=1 -b "$TAICHI_AOT_DEMO_BRANCH" "$TAICHI_AOT_DEMO_URL"

    # Install taichi-python
    pip uninstall -y taichi taichi-nightly || true
    pip install /taichi-wheel/*.whl

    # Build Android Apps
    cd taichi-aot-demo
    ./scripts/build-taichi-android.sh
    ./scripts/build-android.sh
    ./scripts/build-android-app.sh E3_implicit_fem

    run-android-app \
        framework/android/app/build/outputs/apk/debug/E3_implicit_fem-debug.apk \
        org.taichi.aot_demo/android.app.NativeActivity
}

function prepare-unity-build-env {
    cd taichi
    python3 .github/workflows/scripts/build.py android --permissive --write-env=/tmp/ti-aot-env.sh
    . /tmp/ti-aot-env.sh

    # Dependencies
    git clone --reference-if-able /var/lib/git-cache -b "$TAICHI_UNITY_EXAMPLE_BRANCH" "$TAICHI_UNITY_EXAMPLE_URL"

    python3 misc/generate_unity_language_binding.py
    cp c_api/unity/*.cs Taichi-UnityExample/Assets/Taichi/Generated
    CAPI_SO_LOC=$(find . -wholename "**/cmake-build/libtaichi_c_api.so")
    cp $CAPI_SO_LOC Taichi-UnityExample/Assets/Plugins/Android

    export TAICHI_REPO_DIR=$(pwd)

    git clone --reference-if-able /var/lib/git-cache -b "$TAICHI_UNITY2_BRANCH" "$TAICHI_UNITY2_URL"
    mkdir tu2-build
    pushd tu2-build
    cmake ../taichi-unity2 $TAICHI_CMAKE_ARGS
    cmake --build .
    popd
    cp tu2-build/bin/libtaichi_unity.so Taichi-UnityExample/Assets/Plugins/Android

    pushd Taichi-UnityExample
    python3 -m pip uninstall -y taichi taichi-nightly || true
    python3 -m pip install /taichi-wheel/*.whl
    python3 scripts/implicit_fem.cgraph.py --aot
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
    pushd taichi
    python3 .github/workflows/scripts/build.py android --permissive --write-env=/tmp/ti-aot-env.sh
    . /tmp/ti-aot-env.sh
    pip uninstall -y taichi taichi-nightly || true
    pip install /taichi-wheel/*.whl
    sudo chmod 0777 $HOME/.cache
    popd

    rm -rf taichi-aot-demo
    git clone --recursive --jobs=4 --depth=1 -b "$TAICHI_AOT_DEMO_BRANCH" "$TAICHI_AOT_DEMO_URL"
    cd taichi-aot-demo

    . $(pwd)/ci/test_utils.sh

    # Build demos
    build_demos "$TAICHI_CMAKE_ARGS"

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
    python3 .github/workflows/scripts/build.py wheel --permissive --write-env=/tmp/ti-aot-env.sh
    . /tmp/ti-aot-env.sh
    python3 -m pip uninstall -y taichi taichi-nightly || true
    python3 -m pip install dist/*.whl
    popd

    sudo chmod 0777 $HOME/.cache

    rm -rf taichi-aot-demo
    git clone --recursive --jobs=4 --depth=1 -b "$TAICHI_AOT_DEMO_BRANCH" "$TAICHI_AOT_DEMO_URL"
    cd taichi-aot-demo

    python3 -m pip install -r ci/requirements.txt
    python3 ci/run_tests.py -l $TAICHI_C_API_INSTALL_DIR
}

function check-c-api-export-symbols {
    [ ! -z $IN_DOCKER ] && cd taichi

    python3 .github/workflows/scripts/build.py wheel --permissive --write-env=/tmp/ti-aot-env.sh
    . /tmp/ti-aot-env.sh

    LIBTAICHI_C_API=$TAICHI_C_API_INSTALL_DIR/lib/libtaichi_c_api.so

    # T: global functions
    # B: global variables (uninitialized)
    # D: global variables (initialized)
    EXPORT_SYM=" T \| B \| D "

    # Note: this has to be consistent with the version scripts (export_symbol_linux.ld, export_symbol_mac.ld)
    CAPI_SYM=" _\?ti_"
    CAPI_UTILS_SYM=" capi::utils::"

    NUM_LEAK_SYM=$(nm -C --extern-only ${LIBTAICHI_C_API} | grep "${EXPORT_SYM}" | grep -v "${CAPI_SYM}" | grep -v "${CAPI_UTILS_SYM}" | wc -l)
    if [ ${NUM_LEAK_SYM} -gt 0 ]; then
        echo "Following symbols leaked from libtaichi_c_api: "
        nm -C --extern-only ${LIBTAICHI_C_API} | grep "${EXPORT_SYM}" | grep -v "${CAPI_SYM}" | grep -v "${CAPI_UTILS_SYM}"
        exit 1
    fi
}

$1
