#!/bin/bash
set -ex

. $(dirname $0)/common-utils.sh

export PYTHONUNBUFFERED=1

[[ "$IN_DOCKER" == "true" ]] && cd taichi

python3 .github/workflows/scripts/build.py android --write-env=/tmp/ti-env.sh
. /tmp/ti-env.sh

python3 -m pip install dist/*.whl

git clone https://github.com/taichi-dev/taichi_benchmark
cd taichi_benchmark
pip install -r requirements.txt
python run.py --upload-auth $BENCHMARK_UPLOAD_TOKEN
