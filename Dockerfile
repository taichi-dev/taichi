FROM nvidia/cuda:10.0-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libtinfo-dev \
    wget \
    libx11-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    zlib1g-dev \
    xz-utils \
    curl \
    libomp5 \
    libomp-dev

RUN curl -SL http://releases.llvm.org/7.0.1/clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz | tar -xJC . \
    && cp -r clang+llvm-7.0.1-x86_64-linux-gnu-ubuntu-16.04/ /usr/local/clang-7.0.1
ENV LD_LIBRARY_PATH=/usr/local/clang-7.0.1/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/clang-7.0.1/bin:$PATH
RUN ldconfig

ENV CC=/usr/local/clang-7.0.1/bin/clang
ENV CXX=/usr/local/clang-7.0.1/bin/clang++
RUN curl -SL https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/llvm-8.0.1.src.tar.xz | tar -xJC .
RUN cd llvm-8.0.1.src && mkdir build && cd build && cmake .. -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON && make -j `nproc --all` && make install

# install python 3.6 from source since it is not available on ubuntu 16.04's official repo
RUN apt install -y build-essential checkinstall
RUN apt install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
RUN wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tar.xz && \
    tar xvf Python-3.6.0.tar.xz && \
    cd Python-3.6.0/ && \
    ./configure --enable-shared && \
    make altinstall

WORKDIR /
# install python dependencies
RUN curl https://bootstrap.pypa.io/get-pip.py --output ./get-pip.py \
    && python3.6 ./get-pip.py
RUN python3.6 -m pip install twine numpy Pillow scipy pybind11 colorama setuptools astor matplotlib pytest autograd 

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt install g++-7 -y
RUN echo /usr/local/cuda-10.0/compat >> /etc/ld.so.conf.d/cuda-10-0.conf && ldconfig

RUN mkdir /app
WORKDIR /app

ENV SHELL=bash
RUN cd /app && git clone https://github.com/yuanming-hu/taichi.git
RUN mkdir /app/taichi/build
WORKDIR /app/taichi/build
RUN export CUDA_BIN_PATH=/usr/local/cuda-10.0 && \
    cmake .. -DPYTHON_EXECUTABLE=$(which python3.6) -DCUDA_VERSION=10.0 -DTI_WITH_CUDA:BOOL="False" && \
    make -j 15 && \
    ldd libtaichi_core.so

WORKDIR /app/taichi/python
ENV PATH="/app/taichi/bin:$PATH"
ENV TAICHI_REPO_DIR="/app/taichi/"
ENV PYTHONPATH="$TAICHI_REPO_DIR/python:$PYTHONPATH"
RUN ti test

WORKDIR /app
CMD /bin/bash
