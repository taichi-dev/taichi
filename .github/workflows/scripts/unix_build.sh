#!/bin/bash
python3 -m pip uninstall taichi taichi-nightly -y
python3 -m pip install -r requirements_dev.txt
python3 -m pip install -r requirements_test.txt
git fetch origin master

PROJECT_TAGS=""
EXTRA_ARGS=""
if [ $PROJECT_NAME -eq "taichi-nightly" ]; then
    PROJECT_TAGS="egg_info --tag-date"
fi

if [[ $OSTYPE == "linux-"* ]]; then
    EXTRA_ARGS="-p manylinux1_x86_64"
fi
python3 misc/make_changelog.py origin/master ./ True
python3 setup.py $PROJECT_TAGS bdist_wheel $EXTRA_ARGS

export NUM_WHL=`ls dist/*.whl | wc -l`
if [ $NUM_WHL -ne 1 ]; then echo `ERROR: created more than 1 whl.` && exit 1; fi
python3 -m pip install dist/*.whl

rm -f CHANGELOG.md
