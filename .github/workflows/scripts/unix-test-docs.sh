#!/bin/bash
set -ex

. $(dirname $0)/common-utils.sh

export PYTHONUNBUFFERED=1

export TI_CI=1
export LD_LIBRARY_PATH=$PWD/build/:$LD_LIBRARY_PATH
export TI_OFFLINE_CACHE_FILE_PATH=$PWD/.cache/taichi

[[ "$IN_DOCKER" == "true" ]] && cd taichi

setup_python
python3 -m pip install dist/*.whl

export PATH=$PATH:$HOME/.local/bin
python3 -m pip install -r requirements_test.txt

cat docs/cover-in-ci.lst | xargs pytest -v -n 8
