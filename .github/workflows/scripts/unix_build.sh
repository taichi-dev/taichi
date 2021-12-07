#!/bin/bash
set -ex

check_in_docker() {
    # This is a temporary solution to detect in a docker, but it should work
    if [[ $(whoami) == "dev" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

IN_DOCKER=$(check_in_docker)
[[ "$IN_DOCKER" == "true" ]] && cd taichi

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
    if [[ "$IN_DOCKER" == "true" ]]; then
        source $HOME/miniconda/etc/profile.d/conda.sh
        conda activate "$PY"
    fi
    python3 -m pip uninstall taichi taichi-nightly -y
    if [ -z "$GPU_BUILD" ]; then
        python3 -m pip install -r requirements_dev.txt
    fi
}

build() {
    git fetch origin master
    PROJECT_TAGS=""
    EXTRA_ARGS=""
    if [ "$PROJECT_NAME" = "taichi-nightly" ]; then
        PROJECT_TAGS="egg_info --tag-date"
    fi

    if [[ $OSTYPE == "linux-"* ]]; then
        EXTRA_ARGS="-p manylinux1_x86_64"
    fi
    python3 misc/make_changelog.py origin/master ./ True
    python3 setup.py $PROJECT_TAGS bdist_wheel $EXTRA_ARGS
    sccache -s
}

setup_sccache
setup_python
build
cat "$SCCACHE_ERROR_LOG"
NUM_WHL=$(ls dist/*.whl | wc -l)
if [ $NUM_WHL -ne 1 ]; then echo "ERROR: created more than 1 whl." && exit 1; fi

chmod -R 777 "$SCCACHE_DIR"
rm -f python/CHANGELOG.md
