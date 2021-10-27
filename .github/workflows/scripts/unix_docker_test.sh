#!/bin/bash

set -ex

# Parse ARGs
PY=$1
GPU_TEST=$2

source /home/dev/miniconda/etc/profile.d/conda.sh
conda activate $PY

python3 -m pip install ./*.whl

if [[ $GPU_TEST == "OFF" ]]
then
    python3 -m pip install -r requirements_test.txt
fi

export TI_IN_DOCKER=true
ti diagnose
ti changelog

if [[ $GPU_TEST == "OFF" ]]
then
    ti test -vr2 -t2
else
    ti test -vr2 -t2 -k "not ndarray and not torch"
    ti test -vr2 -t1 -k "ndarray or torch"
fi
