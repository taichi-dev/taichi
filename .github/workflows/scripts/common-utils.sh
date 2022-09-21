#!/bin/bash

set -x

setup_python() {
    for conda in miniconda miniconda3 miniforge3; do
        if [[ -d $HOME/$conda ]]; then
            source $HOME/$conda/bin/activate
            conda activate "$PY"
            break
        fi
    done
    python3 -m pip install -U pip
    python3 -m pip uninstall taichi taichi-nightly -y
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

function clear-taichi-offline-cache {
    rm -rf ~/build-cache/dot-cache/taichi  # Clear taichi offline cache
    rm -rf ~/.cache/taichi  # Clear taichi offline cache
}

function prepare-build-cache {
    export CACHE_HOME=${1:-$HOME/build-cache}
    mkdir -p $CACHE_HOME/{dot-cache/pip,dot-gradle,git-cache,sccache/{bin,cache}}
    chmod 0777 $CACHE_HOME/ $CACHE_HOME/* || true
    pushd $CACHE_HOME/git-cache
    if [ ! -d objects ]; then git init --bare; fi
    popd

    clear-taichi-offline-cache

    if [ ! -z $GITHUB_ENV ]; then
        # for bare metal run
        export SCCACHE_ROOT=$CACHE_HOME/sccache
        export GIT_ALTERNATE_OBJECT_DIRECTORIES=$CACHE_HOME/git-cache/objects
        echo SCCACHE_ROOT=$SCCACHE_ROOT >> $GITHUB_ENV
        echo GIT_ALTERNATE_OBJECT_DIRECTORIES=$GIT_ALTERNATE_OBJECT_DIRECTORIES >> $GITHUB_ENV
        echo CACHE_HOME=$CACHE_HOME >> $GITHUB_ENV
    else
        # container run
        true
    fi
}

function fix-build-cache-permission {
    if [ "$(uname -s)" = "Linux" ]; then
        sudo -n chown -R $(id -u):$(id -g) $CACHE_HOME || true
    fi
}

function ci-docker-run {
    ARGS="$@"
    SHOULD_RM="--rm"
    while [[ $# > 0 ]]; do
        case $1 in
            -n | --name)
                shift
                CONTAINER_NAME="$1"
                SHOULD_RM=""
                break
                ;;
        esac
        shift
    done

    if [ ! -z $CONTAINER_NAME ]; then
        docker rm -f $CONTAINER_NAME
    fi

    TI_ENVS=""
    for i in $(env | grep ^TI_); do
        TI_ENVS="$TI_ENVS -e $i"
    done

    CACHE_HOME=${CACHE_HOME:-$HOME/build-cache}

    docker run \
        -i \
        $SHOULD_RM \
        --user dev \
        -e PY \
        -e PROJECT_NAME \
        -e TAICHI_CMAKE_ARGS \
        -e IN_DOCKER=true \
        -e TI_CI=1 \
        $TI_ENVS \
        -e SCCACHE_ROOT=/var/lib/sccache \
        -e CACHE_HOME=/var/lib/cache-home \
        -e GIT_ALTERNATE_OBJECT_DIRECTORIES=/var/lib/git-cache/objects \
        -v $(readlink -f $CACHE_HOME):/var/lib/cache-home \
        -v $(readlink -f $CACHE_HOME/sccache):/var/lib/sccache \
        -v $(readlink -f $CACHE_HOME/git-cache):/var/lib/git-cache \
        -v $(readlink -f $CACHE_HOME/dot-cache):/home/dev/.cache \
        -v $(readlink -f $CACHE_HOME/dot-gradle):/home/dev/.gradle \
        $CI_DOCKER_RUN_EXTRA_ARGS \
        $ARGS
}

function ci-docker-run-gpu {
    for i in {0..9}; do
        if xset -display ":$i" -q >/dev/null 2>&1; then
            break
        fi
    done

    if [ $? -ne 0 ]; then
        echo "No display!"
        exit 1
    fi

    ci-docker-run \
        --runtime=nvidia \
        --gpus 'all,"capabilities=graphics,utility,display,video,compute"' \
        -e DISPLAY=:$i \
        -e GPU_BUILD=ON \
        -e GPU_TEST=ON \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        $@
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
