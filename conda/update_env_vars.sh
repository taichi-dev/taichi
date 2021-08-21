#!/bin/bash

if [ "$CONDA_DEFAULT_ENV" != "taichi-dev" ]; then
    echo "Please run inside taichi-dev conda env"
    exit 1
fi

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
export MY_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
export SCRIPTS_DIR=$MY_DIR/scripts

# installs the activation/deactivation scripts for taichi-dev

# See https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux

cd $CONDA_PREFIX

# install activation script

DST_DIR=./etc/conda/activate.d

mkdir -p $DST_DIR
pushd $DST_DIR
rm -f env_vars.sh
cp $SCRIPTS_DIR/activate_env_vars.sh env_vars.sh
popd
