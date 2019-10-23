# Temporary documentation for this Dockerfile: https://github.com/yuanming-hu/taichi/issues/214
# Contributor: robbertvc and yuanming-hu

FROM nvidia/cuda:10.0-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3.6-venv \
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

# install python dependencies
RUN python3.6 -m pip install \
    astpretty \
    astor \
    pytest \
    pybind11 \
    Pillow \
    numpy \
    scipy \
    distro

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt install g++-7 -y
RUN echo /usr/local/cuda-10.0/compat >> /etc/ld.so.conf.d/cuda-10-0.conf && ldconfig

RUN mkdir /app
WORKDIR /app

ENV SHELL=bash
RUN cd /app && git clone https://github.com/yuanming-hu/taichi.git
RUN cd /app/taichi && python3.6 dev_setup.py
RUN cd /app/taichi && mkdir build && cd build && cmake .. -DPYTHON_EXECUTABLE=python3.6
RUN cd /app/taichi/build && make -j `nproc --all`

WORKDIR /app
CMD /bin/bash