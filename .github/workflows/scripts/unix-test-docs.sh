#!/bin/bash
set -ex

. $(dirname $0)/common-utils.sh

export PYTHONUNBUFFERED=1

export TI_CI=1
export LD_LIBRARY_PATH=$PWD/build/:$LD_LIBRARY_PATH
export TI_OFFLINE_CACHE_FILE_PATH=$PWD/.cache/taichi

[[ "$IN_DOCKER" == "true" ]] && cd taichi

python3 .github/workflows/scripts/build.py --permissive --write-env=/tmp/ti-env.sh
. /tmp/ti-env.sh

python3 -m pip uninstall -y taichi taichi-nightly || true
python3 -m pip install dist/*.whl

export PATH=$PATH:$HOME/.local/bin
python3 -m pip install -r requirements_test.txt
python3 -m pip install torch

cat docs/cover-in-ci.lst | xargs pytest -v -n 4
