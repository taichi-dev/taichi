.. _dev_install:

Developer installation
======================

Note this is for the compiler developers of the Taichi programming language.
End users should use the pip packages instead of building from scratch.
To build with NVIDIA GPU support, CUDA 10.0+ is needed.
This installation guide works for Ubuntu 16.04+ and OS X 10.14+.
For precise build instructions on Windows, please check out `appveyor.yml <https://github.com/taichi-dev/taichi/blob/master/appveyor.yml>`_, which does basically the same thing as the following instructions.

Note that on Linux/OS X, ``clang`` is the only supported compiler for compiling the Taichi compiler. On Windows only MSVC supported.

Installing Depedencies
----------------------

- Make sure you are using Python 3.6/3.7/3.8
- Execute

  .. code-block:: bash

    python3 -m pip install --user setuptools astpretty astor pytest opencv-python pybind11
    python3 -m pip install --user Pillow numpy scipy GitPython yapf colorama psutil autograd

* (If on Ubuntu) Execute ``sudo apt install libtinfo-dev clang-8``. (``clang-7`` should work as well).

* (If on other Linux distributions) Please build clang 8.0.1 from scratch:

  .. code-block:: bash

    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/cfe-8.0.1.src.tar.xz
    tar xvJf cfe-8.0.1.src.tar.xz
    cd cfe-8.0.1.src
    mkdir build
    cd build
    cmake ..
    make -j 8
    sudo make install


- Make sure you have LLVM 8.0.1 built from scratch. To do so:

  .. code-block:: bash

    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/llvm-8.0.1.src.tar.xz
    tar xvJf llvm-8.0.1.src.tar.xz
    cd llvm-8.0.1.src
    mkdir build
    cd build
    cmake .. -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON
    # If you are building on NVIDIA Jetson TX2, use -DLLVM_TARGETS_TO_BUILD="ARM;NVPTX"
    make -j 8
    sudo make install

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
    git submodule update --init --recursive --depth=1
    cd taichi
    mkdir build
    cd build
    cmake ..
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



Prebuilt LLVM for Windows CI
----------------------------

.. code-block:: bash

  cmake .. -G"Visual Studio 15 2017 Win64"  -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64 -DLLVM_BUILD_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=installed

Then use Visual Studio to build. After building the ``INSTALL`` project (under folder "CMakePredefinedTargets"). After build completes, find your LLVM binaries/headers in `build/include`.

Troubleshooting
---------------

- Run with debug mode to see if there's any illegal memory access
- Disable compiler optimizations to quickly confirm that the issue is not cause by optimization
