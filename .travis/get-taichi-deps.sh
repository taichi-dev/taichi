#!/bin/bash
set -e

mkdir -p taichi-llvm
cd taichi-llvm
if [ $TRAVIS_OS_NAME == 'osx' ]; then
	wget https://github.com/yuanming-hu/taichi_assets/releases/download/llvm8/taichi-llvm-8.0.1.zip --retry-on-http-error=403 --waitretry=3 --tries=5
	unzip taichi-llvm-8.0.1.zip
elif [ $TRAVIS_OS_NAME == 'linux' ]; then
	wget https://github.com/yuanming-hu/taichi_assets/releases/download/llvm8/taichi-llvm-8.0.1-linux-x64.zip --retry-on-http-error=403 --waitretry=3 --tries=5
	unzip taichi-llvm-8.0.1-linux-x64.zip
fi
cd ..
