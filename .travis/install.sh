#!/bin/bash

set -e

# use brew & pyenv to build specific python on osx
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "${MATRIX_EVAL}"
    echo "python version: $PYTHON_VERSION"
    brew update > /dev/null
    brew upgrade pyenv
    # use pyenv to build python
    eval "$(pyenv init -)"
    pyenv install $PYTHON_VERSION
    pyenv global $PYTHON_VERSION
    pyenv rehash
elif [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "${MATRIX_EVAL}"
    echo "python version: $PYTHON_VERSION"
    if [[ $PYTHON != 'python3.6' ]]; then
        sudo apt-get install $PYTHON-dev
    fi
    $PYTHON --version
fi
