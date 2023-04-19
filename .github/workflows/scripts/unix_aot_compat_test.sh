#!/bin/bash
set -ex

. $(dirname $0)/common-utils.sh

export PYTHONUNBUFFERED=1

export TAICHI_AOT_FOLDER_PATH="taichi/tests"
export TI_SKIP_VERSION_CHECK=ON
export LD_LIBRARY_PATH=$PWD/build/:$LD_LIBRARY_PATH
export TI_OFFLINE_CACHE_FILE_PATH=$PWD/.cache/taichi

[[ "$IN_DOCKER" == "true" ]] && cd taichi

python3 .github/workflows/scripts/build.py --permissive --write-env=/tmp/ti-env.sh
. /tmp/ti-env.sh

pip3 install -i https://pypi.taichi.graphics/simple/ taichi-nightly
python3 tests/generate_compat_test_modules.py
python3 -m pip uninstall taichi-nightly -y

install_taichi_wheel

python3 tests/run_c_api_compat_test.py
