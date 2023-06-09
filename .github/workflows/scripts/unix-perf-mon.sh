#!/bin/bash
set -ex

. $(dirname $0)/common-utils.sh

export PYTHONUNBUFFERED=1

[[ "$IN_DOCKER" == "true" ]] && cd taichi


python3 .github/workflows/scripts/build.py --permissive --write-env=/tmp/ti-env.sh
. /tmp/ti-env.sh

# TODO: hard code Android NDK path in Docker image, should be handled by build.py
export ANDROID_NDK_ROOT=/android-sdk/ndk-bundle

python -m pip uninstall -y taichi taichi-nightly || true
python -m pip install dist/*.whl

TAG=$(git describe --exact-match --tags 2>/dev/null || true)

git clone --depth=1 https://github.com/taichi-dev/taichi_benchmark

cd taichi_benchmark
pip install -r requirements.txt


if [ "$GITHUB_EVENT_ACTION" == "benchmark-command" ]; then
    python run.py --save ../result.json
    cd ..
    python .github/workflows/scripts/post-benchmark-to-github-pr.py /github-event.json result.json
else
    if [ ! -z "$TAG" ]; then
        MORE_TAGS="--tags type=release,release=$TAG"
    else
        MORE_TAGS=""
    fi
    python run.py --upload-auth $BENCHMARK_UPLOAD_TOKEN $MORE_TAGS
fi
