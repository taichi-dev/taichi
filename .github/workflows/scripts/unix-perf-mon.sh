#!/bin/bash
set -ex

. $(dirname $0)/common-utils.sh

export PYTHONUNBUFFERED=1

[[ "$IN_DOCKER" == "true" ]] && cd taichi

python3 .github/workflows/scripts/build.py --write-env=/tmp/ti-env.sh
. /tmp/ti-env.sh

# TODO: hard code Android NDK path in Docker image, should be handled by build.py
export ANDROID_NDK_ROOT=/android-sdk/ndk-bundle

python -m pip install dist/*.whl

TAG=$(git describe --exact-match --tags 2>/dev/null || true)
if [ ! -z "$TAG" ]; then
    MORE_TAGS="--tags type=release,release=$TAG"
else
    MORE_TAGS=""
fi

git clone https://github.com/taichi-dev/taichi_benchmark
cd taichi_benchmark
pip install -r requirements.txt
python run.py --upload-auth $BENCHMARK_UPLOAD_TOKEN $MORE_TAGS
