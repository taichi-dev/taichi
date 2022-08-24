#!/bin/bash

check_in_docker() {
    # This is a temporary solution to detect in a docker, but it should work
    if [[ $(whoami) == "dev" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

setup_sccache() {
    export SCCACHE_DIR=$(pwd)/sccache_cache
    export SCCACHE_CACHE_SIZE="128M"
    export SCCACHE_LOG=error
    export SCCACHE_ERROR_LOG=$(pwd)/sccache_error.log
    mkdir -p "$SCCACHE_DIR"
    echo "sccache dir: $SCCACHE_DIR"
    ls -la "$SCCACHE_DIR"

    if [[ $OSTYPE == "linux-"* ]]; then
        wget https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz
        tar -xzf sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz
        chmod +x sccache-v0.2.15-x86_64-unknown-linux-musl/sccache
        export PATH=$(pwd)/sccache-v0.2.15-x86_64-unknown-linux-musl:$PATH
    elif [[ $(uname -m) == "arm64" ]]; then
        wget https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-aarch64-apple-darwin.tar.gz
        tar -xzf sccache-v0.2.15-aarch64-apple-darwin.tar.gz
        chmod +x sccache-v0.2.15-aarch64-apple-darwin/sccache
        export PATH=$(pwd)/sccache-v0.2.15-aarch64-apple-darwin:$PATH
    else
        wget https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-apple-darwin.tar.gz
        tar -xzf sccache-v0.2.15-x86_64-apple-darwin.tar.gz
        chmod +x sccache-v0.2.15-x86_64-apple-darwin/sccache
        export PATH=$(pwd)/sccache-v0.2.15-x86_64-apple-darwin:$PATH
    fi
}

setup_python() {
    if [[ "$(check_in_docker)" == "true" ]]; then
        source $HOME/miniconda/etc/profile.d/conda.sh
        conda activate "$PY"
    fi
    python3 -m pip uninstall taichi taichi-nightly -y
    python3 -m pip install -r requirements_dev.txt
}

function setup-sccache-local {
    if [ -z $SCCACHE_ROOT ]; then
        echo "Skipping sccache setup since SCCACHE_ROOT is not set"
        return
    fi

    mkdir -p $SCCACHE_ROOT/{bin,cache}

    export SCCACHE_DIR=$SCCACHE_ROOT/cache
    export SCCACHE_CACHE_SIZE="10G"
    export SCCACHE_LOG=error
    export SCCACHE_ERROR_LOG=$SCCACHE_ROOT/sccache_error.log

    echo "SCCACHE_ROOT: $SCCACHE_ROOT"

    if [ ! -x $SCCACHE_ROOT/bin/sccache ]; then
        if [[ $OSTYPE == "linux-"* ]]; then
            wget https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz
            tar -xzf sccache-v0.2.15-x86_64-unknown-linux-musl.tar.gz
            mv sccache-v0.2.15-x86_64-unknown-linux-musl/* $SCCACHE_ROOT/bin
        elif [[ $(uname -m) == "arm64" ]]; then
            wget https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-aarch64-apple-darwin.tar.gz
            tar -xzf sccache-v0.2.15-aarch64-apple-darwin.tar.gz
            mv sccache-v0.2.15-aarch64-apple-darwin/* $SCCACHE_ROOT/bin
        else
            wget https://github.com/mozilla/sccache/releases/download/v0.2.15/sccache-v0.2.15-x86_64-apple-darwin.tar.gz
            tar -xzf sccache-v0.2.15-x86_64-apple-darwin.tar.gz
            mv sccache-v0.2.15-x86_64-apple-darwin/* $SCCACHE_ROOT/bin
        fi
        chmod +x $SCCACHE_ROOT/bin/sccache
    fi

    export PATH=$SCCACHE_ROOT/bin:$PATH
    export TAICHI_CMAKE_ARGS="$TAICHI_CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
}

function setup-android-ndk-env {
    export ANDROID_NDK_ROOT=/android-sdk/ndk-bundle
    export ANDROID_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake -DANDROID_NATIVE_API_LEVEL=29 -DANDROID_ABI=arm64-v8a"
    export TAICHI_CMAKE_ARGS="$TAICHI_CMAKE_ARGS $ANDROID_CMAKE_ARGS"
    export PATH=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH
}

function grab-android-bot {
    if [ -z "$REDIS_HOST" ]; then
        echo "REDIS_HOST is not set"
        exit 1
    fi

    export BOT_LOCK_COOKIE=$(cat /proc/sys/kernel/random/uuid)
    while true; do
        for bot in $(redis-cli -h $REDIS_HOST --raw lrange android-bots 0 -1); do
            export BOT_LOCK_KEY="android-bot-lock:$bot"
            LOCKED=$(redis-cli -h $REDIS_HOST --raw setnx $BOT_LOCK_KEY $BOT_LOCK_COOKIE)
            if [ $LOCKED -eq 1 ]; then
                redis-cli -h $REDIS_HOST --raw expire android-bot-lock:$bot 300 > /dev/null
                break
            fi
        done

        if [ "$LOCKED" == "1" ]; then
            export BOT=$bot
            break
        fi

        echo "No bots available, retrying..."
        sleep 1
    done
}

function release-android-bot {
    if [ $(redis-cli -h $REDIS_HOST --raw get $BOT_LOCK_KEY) == $BOT_LOCK_COOKIE ]; then
        redis-cli -h $REDIS_HOST --raw del $BOT_LOCK_KEY
    fi
}

function run-android-app {
    APK=$1
    ACTIVITY=$2
    WAIT_TIME=${3:-5}
    grab-android-bot
    /android-sdk/platform-tools/adb connect $BOT
    /android-sdk/platform-tools/adb devices
    /android-sdk/platform-tools/adb install $APK
    sleep 1
    /android-sdk/platform-tools/adb logcat -c
    /android-sdk/platform-tools/adb shell am start $ACTIVITY
    sleep $WAIT_TIME
    /android-sdk/platform-tools/adb logcat -d -v time 'CRASH:E *:F' | tee logcat.log
    /android-sdk/platform-tools/adb shell am force-stop $(echo $ACTIVITY | sed 's#/.*$##g')
    /android-sdk/platform-tools/adb disconnect
    release-android-bot
    if [ -s logcat.log ]; then
        echo "!!!!!!!!!!!!!! Something is wrong !!!!!!!!!!!!!!"
        exit 1
    fi
}
