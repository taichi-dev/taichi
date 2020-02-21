#!/bin/bash

# use brew & pyenv to build specified python
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    export PATH="$HOME/.pyenv/bin:$PATH"
    brew update
    eval "$(pyenv init -)"
    pyenv install $PYENV_VERSION
    pyenv global $PYENV_VERSION
    pyenv rehash
fi
