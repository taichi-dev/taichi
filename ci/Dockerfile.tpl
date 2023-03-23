# +=======================================+
# | PATCHES                               |
# | These should be merged into the image |
# +=======================================+


# ===== END PATCHES ======
SNIPPET mitm-ca
COPY assets/mitm-ca.crt /usr/local/share/ca-certificates/mitm-ca.crt
RUN chmod 644 /usr/local/share/ca-certificates/mitm-ca.crt && \
    update-ca-certificates

SNIPPET mitm-ca-centos
COPY assets/mitm-ca.crt /etc/pki/ca-trust/source/anchors/mitm-ca.crt
RUN chmod 644 /etc/pki/ca-trust/source/anchors/mitm-ca.crt && \
    update-ca-trust

# -------
SNIPPET devtoolset-11
RUN curl -o /etc/yum.repos.d/CentOS-Base.repo http://mirrors.cloud.tencent.com/repo/centos7_base.repo && \
    yum updateinfo && \
    yum install -y epel-release centos-release-scl-rh && \
    yum install -y devtoolset-11-{gcc,gcc-c++} git wget unzip sudo libXrandr which rh-python38 && \
    . /opt/rh/rh-python38/enable && \
    python3 -m pip install -U pip && \
    python3 -m pip install -U cmake && \
    true

ENV PATH="/opt/rh/rh-python38/root/usr/local/bin:/opt/rh/rh-python38/root/usr/bin:$PATH"

# -------
SNIPPET dev-user
RUN useradd -ms /bin/bash dev && \
    usermod -a -G video dev && \
    printf "root ALL=(ALL:ALL) NOPASSWD: ALL\ndev ALL=(ALL:ALL) NOPASSWD: ALL\n" > /etc/sudoers && \
    true
WORKDIR /home/dev
ENV LANG="C.UTF-8"
USER dev

# -------
SNIPPET debian-addons
ENV DEBIAN_FRONTEND=noninteractive
RUN set -x && \
    sed -i 's/archive.ubuntu.com/mirrors.tencent.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tencent.com/g' /etc/apt/sources.list && \
    apt update && \
    apt upgrade -y && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 15CF4D18AF4F7421 && \
    echo 'deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-15 main' > /etc/apt/sources.list.d/llvm-15.list && \
    apt update && \
    apt install -y build-essential clang-15 clang-tidy-15 llvm-15 lld-15 gcc-11 g++-11 curl wget sudo python3-pip git unzip && \
    python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -U cmake && \
    cd /usr && \
    curl -L https://github.com/rui314/mold/releases/download/v1.10.1/mold-1.10.1-x86_64-linux.tar.gz | tar -xz --strip-components=1 && \
    rm /usr/bin/ld && ln -sf /usr/bin/mold /usr/bin/ld && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/clang-15 /usr/bin/clang && \
    ln -sf /usr/bin/clang++-15 /usr/bin/clang++ && \
    ln -sf /usr/bin/lld-15 /usr/bin/lld && \
    ln -sf /usr/bin/ld.lld-15 /usr/bin/ld.lld && \
    true

# mold version is not special, just the latest one at the time of writing
# lld is for AMDGPU backend, C++ code runs it.

ENV CC="/usr/bin/clang" \
    CXX="/usr/bin/clang++"

# -------
SNIPPET debian-addons-test
ENV DEBIAN_FRONTEND=noninteractive
RUN set -x && \
    sed -i 's/archive.ubuntu.com/mirrors.tencent.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tencent.com/g' /etc/apt/sources.list && \
    apt update && apt upgrade -y && \
    apt install -y \
        curl wget sudo python3 python3-distutils git unzip \
        libvulkan1 qt5-default libglfw3 ffmpeg \
        && \
    curl -sSL https://bootstrap.pypa.io/pip/3.6/get-pip.py | python3 - && \
    python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists && \
    true

# -------
FROM ubuntu:18.04 AS build-cpu
USE debian-addons
USE mitm-ca
RUN set -x && \
    apt update && apt install -y libtinfo-dev libx11-xcb-dev && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists && \
    true

USE dev-user
BUILD build-cpu AS registry.botmaster.tgr/taichi-build-cpu:__TIME__

# -------
FROM ubuntu:18.04 AS test-cpu
USE debian-addons-test
USE mitm-ca
USE dev-user
BUILD test-cpu AS registry.botmaster.tgr/taichi-test-cpu:__TIME__

# -------
SNIPPET gpu-build-image-deps
RUN set -x && \
    apt update && apt install -y \
        libtinfo-dev libx11-xcb-dev \
        ffmpeg libxrandr-dev \
        libxinerama-dev libxcursor-dev \
        libxi-dev libglu1-mesa-dev \
        freeglut3-dev mesa-common-dev \
        libssl-dev libglm-dev \
        libxcb-keysyms1-dev libxcb-dri3-dev \
        libxcb-randr0-dev libxcb-ewmh-dev \
        libpng-dev libmirclient-dev \
        libwayland-dev bison \
        liblz4-dev libzstd-dev \
        qt5-default libglfw3 \
        libglfw3-dev libjpeg-dev \
        && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists && \
    true

# -------
FROM rocm/dev-ubuntu-18.04:5.2 AS build-amdgpu
USE debian-addons
USE mitm-ca
USE gpu-build-image-deps
USE dev-user
RUN sudo addgroup --gid 109 render
BUILD build-amdgpu AS registry.botmaster.tgr/taichi-build-amdgpu:__TIME__
BUILD build-amdgpu AS registry.botmaster.tgr/taichi-test-amdgpu:__TIME__

# -------
# Use 11.2 instead of 11.4 to avoid forward compatibility issue on Nvidia driver 460
FROM nvidia/cudagl:11.2.2-devel-ubuntu18.04 AS build-cuda
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
USE debian-addons
USE mitm-ca
USE gpu-build-image-deps
COPY assets/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
# Remove mesa EGL driver, which interferes with the propritary NVIDIA drivers
RUN rm -f /usr/lib/x86_64-linux-gnu/libEGL_mesa*
USE dev-user
BUILD build-cuda AS registry.botmaster.tgr/taichi-build-cuda:__TIME__

# -------
# Use 11.2 instead of 11.4 to avoid forward compatibility issue on Nvidia driver 460
FROM nvidia/cudagl:11.2.2-devel-ubuntu18.04 AS test-cuda
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
USE debian-addons-test
USE mitm-ca
COPY assets/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
# Remove mesa EGL driver, which interferes with the propritary NVIDIA drivers
RUN rm -f /usr/lib/x86_64-linux-gnu/libEGL_mesa*
USE dev-user
BUILD test-cuda AS registry.botmaster.tgr/taichi-test-cuda:__TIME__

# -------
FROM ubuntu:18.04 AS android-sdk
RUN set -x && \
    apt-get update && apt-get -y install wget unzip openjdk-11-jdk redis-tools && \
    mkdir /android-sdk && \
    wget -O clt.zip https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip && \
    unzip clt.zip && rm clt.zip && \
    mv cmdline-tools /android-sdk && \
    yes | /android-sdk/cmdline-tools/bin/sdkmanager --sdk_root=/android-sdk --install \
        'ndk-bundle' \
        'build-tools;30.0.3' \
        'build-tools;33.0.0' \
        'cmake;3.10.2.4988404' \
        'platforms;android-30' \
        'platforms;android-33' \
        'platform-tools' \
        'patcher;v4' \
        emulator \
        && \
    chown -R 1000:1000 /android-sdk && \
    true

FROM build-cuda AS build-android
USER root
ENV ANDROID_SDK_ROOT=/android-sdk
RUN set -x && \
    rm /etc/apt/sources.list.d/* && apt-get update && apt-get -y install wget unzip openjdk-11-jdk redis-tools && \
    true

COPY --from=android-sdk /android-sdk /android-sdk
USER dev
RUN set -x && \
    curl http://172.16.5.1:9000/misc/dot-android.tgz | tar -xz && \
    chown -R dev:dev /home/dev/.android && \
    true

BUILD build-android AS registry.botmaster.tgr/taichi-build-android:__TIME__
BUILD build-android AS registry.botmaster.tgr/taichi-test-android:__TIME__

# ------- intermediate
SNIPPET build-llvm
RUN set -x && \
    . /opt/rh/devtoolset-11/enable && \
    cd /llvm-project && \
    rm -rf build && mkdir -p build && cd build && \
    cmake ../llvm \
        -DLLVM_ENABLE_RTTI:BOOL=ON \
        -DBUILD_SHARED_LIBS:BOOL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_ENABLE_PROJECTS=clang \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -G "Unix Makefiles" \
        && \
    make -j$(nproc) && \
    true

RUN set -x && \
    . /opt/rh/devtoolset-11/enable && \
    cd /llvm-project/build && \
    DESTDIR=__INSTALL__ cmake -P tools/cmake_install.cmake && \
    true

FROM centos:7 AS clang-for-centos
USE devtoolset-11

RUN set -x && \
    cd / && \
    git clone --depth=1 --branch=release/15.x https://github.com/llvm/llvm-project && \
    true

USE build-llvm __INSTALL__=/loop1
ENV CC="/loop1/usr/bin/clang" CXX="/loop1/usr/bin/clang++"
USE build-llvm __INSTALL__=/prepare/clang

# -------
SNIPPET manylinux2014-addons
USE devtoolset-11
COPY --from=clang-for-centos /prepare/clang /
RUN set -x && \
    mkdir -p /usr/lib/gcc/x86_64-redhat-linux && \
    ln -s \
        /opt/rh/devtoolset-11/root/usr/lib/gcc/x86_64-redhat-linux/11 \
        /usr/lib/gcc/x86_64-redhat-linux/11 \
        && \
    clang++ -v && \
    true

FROM centos:7 AS manylinux2014
USE manylinux2014-addons
USE mitm-ca-centos
USE dev-user
BUILD manylinux2014 AS registry.botmaster.tgr/taichi-build-manylinux2014:__TIME__
BUILD manylinux2014 AS registry.botmaster.tgr/taichi-test-manylinux2014:__TIME__

FROM nvidia/cudagl:11.2.2-devel-centos7 AS manylinux2014-cuda
USE manylinux2014-addons
USE mitm-ca-centos
USE dev-user
BUILD manylinux2014-cuda AS registry.botmaster.tgr/taichi-build-manylinux2014-cuda:__TIME__
BUILD manylinux2014-cuda AS registry.botmaster.tgr/taichi-test-manylinux2014-cuda:__TIME__

# -------
# vim: ft=Dockerfile:
