# Taichi Dockerfile for development
FROM nvidia/cudagl:11.2.2-devel-ubuntu20.04
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
                       unzip \
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
# Make sure this URL gets updated each time there is a new prebuilt bin release
RUN wget https://github.com/taichi-dev/taichi_assets/releases/download/llvm10_linux_patch2/taichi-llvm-10.0.0-linux.zip
RUN unzip taichi-llvm-10.0.0-linux.zip && \
    rm taichi-llvm-10.0.0-linux.zip
ENV PATH="/taichi-llvm-10.0.0-linux/bin:$PATH"


# Setting up Vulkan SDK
# References
# [1] https://github.com/edowson/docker-nvidia-vulkan
# [2] https://gitlab.com/nvidia/container-images/vulkan/-/tree/master/docker
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
ENV CC="clang-10"
ENV CXX="clang++-10"
WORKDIR /taichi-dev
# Prevent docker caching when head changes
ADD https://api.github.com/repos/taichi-dev/taichi/git/refs/heads/master version.json
RUN git clone --recursive https://github.com/taichi-dev/taichi --branch=master
WORKDIR /taichi-dev/taichi
RUN python3 -m pip install --user -r requirements_dev.txt
# Update Torch version, otherwise cuda tests fail. See #2969.
RUN python3 -m pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON" python3 setup.py develop --user
# Show ELF info
RUN ldd build/libtaichi_core.so
RUN strings build/libtaichi_core.so | grep GLIBC

# Link Taichi source repo to Python Path
ENV PATH="/taichi-dev/taichi/bin:$PATH"
ENV TAICHI_REPO_DIR="/taichi-dev/taichi/"
ENV PYTHONPATH="$TAICHI_REPO_DIR/python:$PYTHONPATH"
ENV LANG="C.UTF-8"

# Add Docker specific ENV
ENV TI_IN_DOCKER=true
WORKDIR /taichi-dev/taichi
CMD /bin/bash
