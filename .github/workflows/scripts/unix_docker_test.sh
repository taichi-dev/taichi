#!/bin/bash

set -ex

# Parse ARGs
PY=$1
GPU_TEST=$2
TI_WANTED_ARCHS=$3

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
echo wanted archs: $TI_WANTED_ARCHS

if [[ $GPU_TEST == "OFF" ]]
then
    python tests/run_tests.py -vr2 -t2 -a "$TI_WANTED_ARCHS"
else
    python tests/run_tests.py -vr2 -t2 -k "not ndarray and not torch" -a "$TI_WANTED_ARCHS"
    python tests/run_tests.py -vr2 -t1 -k "ndarray or torch" -a "$TI_WANTED_ARCHS"
fi
