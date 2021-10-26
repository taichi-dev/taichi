#!/bin/bash

set -ex

# Parse ARGs
PY=$1
GPU_TEST=$2

source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

python3 -m pip install ./*.whl
[[ $GPU_TEST == "OFF" ]] && python3 -m pip install -r requirements_test.txt

export TI_IN_DOCKER=true
ti diagnose
ti changelog

[[ $GPU_TEST == "OFF" ]] && ti test -vr2 -t2

[[ $GPU_TEST == "ON" ]] && ti test -vr2 -t2 -k "not ndarray and not torch"
[[ $GPU_TEST == "ON" ]] && ti test -vr2 -t1 -k "ndarray or torch"
