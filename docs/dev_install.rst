.. _dev_install:

Developer installation
======================

Note this is for the compiler developers of the Taichi programming language.
End users should use the pip packages instead of building from source.
To build with NVIDIA GPU support, CUDA 10.0+ is needed.
This installation guide works for Ubuntu 16.04+ and OS X 10.14+.
For precise build instructions on Windows, please check out `appveyor.yml <https://github.com/taichi-dev/taichi/blob/master/appveyor.yml>`_, which does basically the same thing as the following instructions. We use MSBUILD.exe to build the generated project. Please note that Windows could have multiple instances of MSBUILD.exe shipped with different products. Please make sure you add the path for MSBUILD.exe within your MSVS directory and make it a higher priority (for instance than the one shipped with .NET).

Note that on Linux/OS X, ``clang`` is the only supported compiler for compiling the Taichi compiler. On Windows only MSVC supported.

Installing Dependencies
-----------------------

- Make sure you are using Python 3.6/3.7/3.8
- Install Python dependencies:

  .. code-block:: bash

    python3 -m pip install --user setuptools astpretty astor pybind11 Pillow dill
    python3 -m pip install --user pytest pytest-rerunfailures pytest-xdist yapf
    python3 -m pip install --user numpy GitPython coverage colorama autograd

- Make sure you have ``clang`` with version >= 7

  * On Windows: Download ``clang-8`` via `this link <https://releases.llvm.org/8.0.0/LLVM-8.0.0-win64.exe>`_.
    Make sure you add the ``bin`` folder containing ``clang.exe`` to the ``PATH`` environment variable.

  * On OS X: you don't need to do anything.

  * On Ubuntu, execute ``sudo apt install libtinfo-dev clang-8``.

  * On other Linux distributions, please build clang 8.0.1 from source:

    .. code-block:: bash

        wget https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/cfe-8.0.1.src.tar.xz
        tar xvJf cfe-8.0.1.src.tar.xz
        cd cfe-8.0.1.src
        mkdir build
        cd build
        cmake ..
        make -j 8
        sudo make install


- Make sure you have LLVM 8.0.1/10.0.0. Note that Taichi uses a customized LLVM so the pre-built binaries from the LLVM official website or other sources probably doesn't work.
  Here we provide LLVM 8.0.1 binaries customized for Taichi, which may or may not work depending on your system environment:
  `Linux <https://github.com/yuanming-hu/taichi_assets/releases/download/llvm8/taichi-llvm-8.0.1-linux-x64.zip>`_,
  `OS X <https://github.com/yuanming-hu/taichi_assets/releases/download/llvm8/taichi-llvm-8.0.1.zip>`_,
  `Windows <https://github.com/yuanming-hu/taichi_assets/releases/download/llvm8/taichi-llvm-8.0.1-msvc2017.zip>`_.

   If the downloaded LLVM does not work, please build from source:

  * On Linux or OS X:

      .. code-block:: bash

        wget https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/llvm-8.0.1.src.tar.xz
        tar xvJf llvm-8.0.1.src.tar.xz
        cd llvm-8.0.1.src
        # For LLVM 10.0.0:
        #     wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/llvm-10.0.0.src.tar.xz
        #     tar xvJf llvm-10.0.0.src.tar.xz
        #     cd llvm-10.0.0.src
        mkdir build
        cd build
        cmake .. -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON
        # If you are building on NVIDIA Jetson TX2, use -DLLVM_TARGETS_TO_BUILD="ARM;NVPTX"

        make -j 8
        sudo make install

        # Check your LLVM installation
        llvm-config --version # You should get 8.0.1 or 10.0.0

  * On Windows:

    .. code-block:: bash

      # LLVM 8.0.1 + MSVC 2017
      cmake .. -G"Visual Studio 15 2017 Win64"  -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64 -DLLVM_BUILD_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=installed

      # LLVM 10.0.0 + MSVC 2019
      cmake .. -G"Visual Studio 16 2019" -A x64  -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64 -DLLVM_BUILD_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=installed

    - Then open ``LLVM.sln`` and use Visual Studio 2017+ to build.
    - Please make sure you are using the ``Release`` configuration. After building the ``INSTALL`` project (under folder ``CMakePredefinedTargets`` in the Solution Explorer window).
    - If you use MSVC 2019, **make sure you use C++17** for the ``INSTALL`` project.
    - After the build is complete, find your LLVM binaries and headers in ``build/installed``.

    Please add ``build/installed/bin`` to ``PATH``.
    Later, when you build Taichi using ``CMake``, set ``LLVM_DIR`` to ``build/installed/lib/cmake/llvm``.

- On Windows, if you use the pre-built LLVM for Taichi, please add ``$LLVM_FOLDER/bin`` to ``PATH``.
  Later, when you build Taichi using ``CMake``, set ``LLVM_DIR`` to ``$LLVM_FOLDER/lib/cmake/llvm``.


Setting up CUDA (optional)
--------------------------

If you don't have CUDA, go to `this website <https://developer.nvidia.com/cuda-downloads>`_ and download the installer.

- To check if CUDA is installed, run ``nvcc --version`` or ``cat /usr/local/cuda/version.txt``.
- On **Ubuntu** we recommend choosing ``deb (local)`` as **Installer Type**.
- On **Arch Linux**, you can easily install CUDA via ``pacman -S cuda`` without downloading the installer manually.


Setting up Taichi for development
---------------------------------

- Clone the taichi repo **recursively**, and build:

  .. code-block:: bash

    git clone https://github.com/taichi-dev/taichi --depth=1 --branch=master
    cd taichi
    git submodule update --init --recursive --depth=1
    mkdir build
    cd build
    cmake ..
    # if you do not set clang as the default compiler
    # use the line below:
    #   cmake .. -DCMAKE_CXX_COMPILER=clang-8
    #
    # Alternatively, if you would like to set clang as the default compiler
    # On Unix CMake honors environment variables $CC and $CXX upon deciding which C and C++ compilers to use
    #
    # if you are building with CUDA 10.0, use the line below:
    #   cmake .. -DCUDA_VERSION=10.0 -DTI_WITH_CUDA:BOOL=True
    make -j 8

- Add the following script to your ``~/.bashrc``:

  .. code-block:: bash

    export TAICHI_REPO_DIR=/home/XXX/taichi  # Path to your taichi repository
    export PYTHONPATH=$TAICHI_REPO_DIR/python/:$PYTHONPATH
    export PATH=$TAICHI_REPO_DIR/bin/:$PATH
    # export PATH=/opt/llvm/bin:$PATH # Uncomment if your llvm-8 or clang-8 is in /opt

- Execute ``source ~/.bashrc`` to reload shell config.
- Execute ``python3 -m taichi test`` to run all the tests. It may take up to 5 minutes to run all tests.
- Check out ``examples`` for runnable examples. Run them with ``python3``.

Docker
------
For those who prefer to use Docker, we also provide a Dockerfile which helps
setup the Taichi development environment with CUDA support based on Ubuntu docker image.

.. note::
    In order to follow the instructions in this section, please make sure you have the
    `Docker DeskTop (or Engine for Linux) <https://www.docker.com/products/docker-desktop>`_ installed and set up
    properly.

Build the Docker Image
**********************
From within the root directory of the taichi Git repository, execute ``docker build -t taichi:master .`` to build a
Docker image based off the local master branch tagged with *master*. Since this builds the image from source, please
expect up to 40 mins build time if you don't have cached Docker image layers.

.. note::
    In order to save the time on building Docker images, you could always visit our `Docker Hub repository <https://hub.docker.com/r/taichidev/taichi>`_ and pull the
    versions of pre-built images you would like to use. Currently the builds are triggered per taichi Github release.

    For example, to pull a image built from release v0.6.17, run ``docker pull taichidev/taichi:v0.6.17``

Use Docker Image on macOS (cpu only)
************************************
1. Make sure ``XQuartz`` and ``socat`` are installed:

.. code-block:: bash

    brew cask install xquartz
    brew install socat

2. Temporally disable the xhost access-control: ``xhost +``
3. Start the Docker container with ``docker run -it -e DISPLAY=$(ipconfig getifaddr en0):0 taichidev/taichi:v0.6.17``
4. Do whatever you want within the container, e.g. you could run tests or an example, try: ``ti test`` or ``ti example mpm88``
5. Exit from the container with ``exit`` or ``ctrl+D``
6. [To keep your xhost safe] Re-enable the xhost access-control: ``xhost -``

Use Docker Image on Ubuntu (with CUDA support)
**********************************************
1. Make sure your host machine has CUDA properly installed and configured. Usually you could verify it by running ``nvidia-smi``
2. Make sure ` NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ is properly installed:

.. code-block:: bash

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker

3. Make sure ``xorg`` is installed: ``sudo apt-get install xorg``
4. Temporally disable the xhost access-control: ``xhost +``
5. Start the Docker container with ``sudo docker run -it --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix taichidev/taichi:v0.6.17``
6. Do whatever you want within the container, e.g. you could run tests or an example, try: ``ti test`` or ``ti example mpm88``
7. Exit from the container with ``exit`` or ``ctrl+D``
8. [To keep your xhost safe] Re-enable the xhost access-control: ``xhost -``

.. warning::
    The nature of Docker container determines that no changes to the file system on the container could be preserved
    once you exit from the container. If you want to use Docker as a persistent development environment, we recommend
    you `mount the taichi Git repository to the container as a volume <https://docs.docker.com/storage/volumes/>`_ and set the Python path to the mounted directory.

Troubleshooting
---------------

- Run with debug mode to see if there's any illegal memory access
- Disable compiler optimizations to quickly confirm that the issue is not cause by optimization
