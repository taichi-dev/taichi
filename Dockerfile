# Taichi Dockerfile for development
FROM nvidia/cuda:10.0-devel-ubuntu18.04

LABEL maintainer="https://github.com/taichi-dev"

# This installs Python 3.6.9 by default. Once the
# docker image is upgraded to Ubuntu 20.04 this will
# install Python 3.8 by default
RUN apt-get update && \
    apt-get install -y software-properties-common \
                       python3-pip \
                       libtinfo-dev \
                       clang-8 \
                       wget \
                       git \
                       libx11-dev \
                       libxrandr-dev \
                       libxinerama-dev \
                       libxcursor-dev \
                       libxi-dev \
                       libglu1-mesa-dev \
                       freeglut3-dev \
                       mesa-common-dev \
                       libtinfo5 \
                       build-essential \
                       libssl-dev

# Install Taichi's Python dependencies
RUN python3 -m pip install --user setuptools astor pybind11 pylint sourceinspect
RUN python3 -m pip install --user pytest pytest-rerunfailures pytest-xdist yapf
RUN python3 -m pip install --user numpy GitPython coverage colorama autograd

# Install the latest version of CMAKE v3.20.2 from source
RUN apt purge --auto-remove cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz
RUN tar -zxvf cmake-3.20.2.tar.gz
RUN cd cmake-3.20.2
WORKDIR /cmake-3.20.2
RUN ./bootstrap
RUN make -j 8
RUN make install

# Intall LLVM 10
WORKDIR /
ENV CC=/usr/bin/clang-8
ENV CXX=/usr/bin/clang++-8
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/llvm-10.0.0.src.tar.xz
RUN tar xvJf llvm-10.0.0.src.tar.xz
RUN cd llvm-10.0.0.src && mkdir build
WORKDIR /llvm-10.0.0.src/build
RUN cmake .. -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON
RUN make -j 8
RUN make install

# Install Taichi from source
WORKDIR /taichi-dev
RUN git clone https://github.com/taichi-dev/taichi --depth=1 --branch=master
RUN cd taichi && \
    git submodule update --init --recursive --depth=1 && \
    mkdir build
WORKDIR /taichi-dev/taichi/build
RUN cmake .. -DPYTHON_EXECUTABLE=python3 -DTI_WITH_CUDA:BOOL=True -DTI_WITH_CC:BOOL=ON
RUN make -j 8

# Link Taichi source repo to Python Path
ENV PATH="/taichi-dev/taichi/bin:$PATH"
ENV TAICHI_REPO_DIR="/taichi-dev/taichi/"
ENV PYTHONPATH="$TAICHI_REPO_DIR/python:$PYTHONPATH"
ENV LANG="C.UTF-8"

# Add Docker specific ENV
ENV TI_IN_DOCKER=true
WORKDIR /taichi-dev/taichi
CMD /bin/bash
