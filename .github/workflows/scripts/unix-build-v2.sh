#!/bin/bash

# TODO: replace unix_build.sh
# currently only used in android build job
set -ex

. $(dirname $0)/common-utils.sh

IN_DOCKER=$(check_in_docker)
[[ "$IN_DOCKER" == "true" ]] && cd taichi

build_taichi_wheel() {
    git fetch origin master --tags
    PROJECT_TAGS=""
    EXTRA_ARGS=""
    if [ "$PROJECT_NAME" = "taichi-nightly" ]; then
        PROJECT_TAGS="egg_info --tag-date"
    fi

    if [[ $OSTYPE == "linux-"* ]]; then
        if [ -f /etc/centos-release ] ; then
            EXTRA_ARGS="-p manylinux2014_x86_64"
        else
            EXTRA_ARGS="-p manylinux_2_27_x86_64"
        fi
    fi
    python3 misc/make_changelog.py --ver origin/master --repo_dir ./ --save

    python3 setup.py $PROJECT_TAGS bdist_wheel $EXTRA_ARGS
    sccache -s
}

setup-sccache-local
setup_python

build_taichi_wheel
NUM_WHL=$(ls dist/*.whl | wc -l)
if [ $NUM_WHL -ne 1 ]; then echo "ERROR: created more than 1 whl." && exit 1; fi

chmod -R 777 "$SCCACHE_DIR"
rm -f python/CHANGELOG.md
