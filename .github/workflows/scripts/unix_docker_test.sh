#!/bin/bash

set -ex

# Parse ARGs
PY=$1
GPU_TEST=$2

source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

python3 -m pip install ./taichi.whl

export TI_IN_DOCKER=true
python3 examples/algorithm/laplace.py
ti diagnose
ti changelog

[ -z $GPU_TEST ] && ti test -vr2 -t2

[ -z $GPU_TEST ] || ti test -vr2 -t2 -k "not ndarray and not torch"
[ -z $GPU_TEST ] || ti test -vr2 -t1 -k "ndarray or torch"
