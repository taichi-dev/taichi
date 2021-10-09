# Taichi Dockerfile for development
FROM nvidia/cudagl:11.4.1-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="https://github.com/taichi-dev"

# Ubuntu 20.04 installs Python 3.8 by default
RUN apt-get update && \
    apt-get install -y software-properties-common \
                       python3-pip \
                       python-is-python3 \
                       libtinfo-dev \
                       clang-10 \
                       wget \
                       git \
                       libxrandr-dev \
                       libxinerama-dev \
                       libxcursor-dev \
                       libxi-dev \
                       libglu1-mesa-dev \
                       freeglut3-dev \
                       mesa-common-dev \
                       libssl-dev \
                       libglm-dev \
                       libxcb-keysyms1-dev \
                       libxcb-dri3-dev \
                       libxcb-randr0-dev \
                       libxcb-ewmh-dev \
                       libpng-dev \
                       g++-multilib \
                       libmirclient-dev \
                       libwayland-dev \
                       bison \
                       libx11-xcb-dev \
                       liblz4-dev \
                       libzstd-dev \
                       qt5-default \
                       libglfw3 \
                       libglfw3-dev \
                       vulkan-tools \
                       libvulkan-dev \
                       vulkan-validationlayers-dev

# Install the latest version of CMAKE v3.20.5 from source
WORKDIR /
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5-linux-x86_64.tar.gz
RUN tar xf cmake-3.20.5-linux-x86_64.tar.gz && \
    rm cmake-3.20.5-linux-x86_64.tar.gz
ENV PATH="/cmake-3.20.5-linux-x86_64/bin:$PATH"


# Intall LLVM 10
WORKDIR /
ENV CC=/usr/bin/clang-10
ENV CXX=/usr/bin/clang++-10
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/llvm-10.0.0.src.tar.xz
RUN tar xvJf llvm-10.0.0.src.tar.xz && \
    rm llvm-10.0.0.src.tar.xz
RUN cd llvm-10.0.0.src && mkdir build
WORKDIR /llvm-10.0.0.src/build
RUN cmake .. -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON
RUN make -j 8
RUN make install

# Setting up Vulkan sdk
WORKDIR /vulkan
RUN wget https://sdk.lunarg.com/sdk/download/1.2.189.0/linux/vulkansdk-linux-x86_64-1.2.189.0.tar.gz
RUN tar xf vulkansdk-linux-x86_64-1.2.189.0.tar.gz && \
    rm vulkansdk-linux-x86_64-1.2.189.0.tar.gz
# Locate Vulkan components
ENV VULKAN_SDK="/vulkan/1.2.189.0/x86_64"
ENV PATH="$VULKAN_SDK/bin:$PATH"
ENV LD_LIBRARY_PATH="$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENV VK_LAYER_PATH="$VULKAN_SDK/etc/vulkan/explicit_layer.d"
WORKDIR /usr/share/vulkan/icd.d
COPY ci/vulkan/icd.d/nvidia_icd.json nvidia_icd.json

# Install Taichi from source
WORKDIR /taichi-dev
RUN git clone https://github.com/taichi-dev/taichi --depth=1 --branch=master
RUN cd taichi && \
    git submodule update --init --recursive --depth=1
WORKDIR /taichi-dev/taichi
RUN python3 -m pip install --user -r requirements_dev.txt
# TODO, otherwise cuda test fails. See #2969
RUN python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON" python3 setup.py develop --user

# Link Taichi source repo to Python Path
ENV PATH="/taichi-dev/taichi/bin:$PATH"
ENV TAICHI_REPO_DIR="/taichi-dev/taichi/"
ENV PYTHONPATH="$TAICHI_REPO_DIR/python:$PYTHONPATH"
ENV LANG="C.UTF-8"

# Add Docker specific ENV
ENV TI_IN_DOCKER=true
WORKDIR /taichi-dev/taichi
CMD /bin/bash
