# Taichi Dockerfile for development
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

ARG PYTHON
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
ENV DEBIAN_FRONTEND=noninteractive

LABEL maintainer="https://github.com/taichi-dev"

RUN apt-get update && \
    apt-get install -y software-properties-common \
                       $PYTHON \
                       python3-pip \
                       ${PYTHON}-dev\
                       libtinfo-dev \
                       clang-10 \
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
                       build-essential \
                       libssl-dev \
                       libidn11-dev \
                       libz-dev \
                       unzip


# Install the latest version of CMAKE v3.20.2 from source
WORKDIR /
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5-linux-x86_64.tar.gz
RUN tar xf cmake-3.20.5-linux-x86_64.tar.gz
ENV PATH="/cmake-3.20.5-linux-x86_64/bin:$PATH"

# Intall LLVM 10
WORKDIR /
# Make sure this URL gets updated each time there is a new prebuilt bin release
RUN wget https://github.com/taichi-dev/taichi_assets/releases/download/llvm10_linux_patch2/taichi-llvm-10.0.0-linux.zip
RUN unzip taichi-llvm-10.0.0-linux.zip
ENV PATH="/taichi-llvm-10.0.0-linux/bin:$PATH"

# Install Taichi from source
ENV CC="clang-10"
ENV CXX="clang++-10"
WORKDIR /taichi-dev

RUN $PYTHON -m pip install cmake colorama coverage numpy Pillow pybind11 GitPython yapf==0.31.0 distro autograd astor sourceinspect pytest pytest-xdist pytest-rerunfailures pytest-cov

# Install Vulkan
RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.2.182-bionic.list http://packages.lunarg.com/vulkan/1.2.182/lunarg-vulkan-1.2.182-bionic.list
RUN apt update
RUN apt install -y vulkan-sdk

# Prevent docker caching when head changes
ADD https://api.github.com/repos/taichi-dev/taichi/git/refs/heads/master version.json
# RUN git clone https://github.com/taichi-dev/taichi --branch=master
ADD . /taichi

WORKDIR /taichi
# # Install Taichi's Python dependencies
# RUN $PYTHON -m pip install --user -r requirements_dev.txt
# # Build Taichi wheel from source
# RUN git submodule update --init --recursive --depth=1
# WORKDIR python/
ENV TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON"
# RUN $PYTHON build.py build
# WORKDIR ../
# RUN $PYTHON -m pip install dist/*.whl
RUN $PYTHON setup.py develop

# # Link Taichi source repo to Python Path
ENV LANG="C.UTF-8"

# # Show ELF info
RUN ldd build/libtaichi_core.so
RUN strings build/libtaichi_core.so | grep GLIBC

# # Install twine and upload project to pypi.
# RUN $PYTHON -m pip install --user twine
RUN $PYTHON -m pytest tests/python -s -k "not ndarray and not torch"
