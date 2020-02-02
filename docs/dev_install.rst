.. _dev_install:

Developer installation
=====================================================

Note this is for the compiler developers of the Taichi programming language.
End users should use the pip packages instead of building from scratch.
To build with NVIDIA GPU support, CUDA 10.0+ is needed.
This installation guide works for Ubuntu 16.04+ and OS X 10.14+.

- Make sure you are using Python 3.6/3.7
- Execute

  .. code-block:: bash

    python3 -m pip install --user setuptools astpretty astor pytest opencv-python pybind11
    python3 -m pip install --user Pillow numpy scipy GitPython yapf colorama psutil autograd

- (If on Ubuntu) Execute ``sudo apt install libtinfo-dev clang-7`` .
- Make sure you have LLVM 8.0.1 built from scratch (`Download <https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/llvm-8.0.1.src.tar.xz>`_). To do so, download and unzip the llvm source, move to the llvm folder, and execute

  .. code-block:: bash

    mkdir build
    cd build
    cmake .. -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON
    make -j 8
    sudo make install

- Clone the taichi repo, and then

  .. code-block:: bash

    cd taichi
    mkdir build
    cd build
    cmake ..
    # if you are building with CUDA, say, 10.0, then please use "cmake .. -DCUDA_VERSION=10.0 -DTLANG_WITH_CUDA:BOOL=True"
    make -j 8

- Add the following to your ``~/.bashrc`` (or ``~/.zshrc`` if you use ``zsh``)

  .. code-block:: bash

    export TAICHI_REPO_DIR=/home/XXX/taichi  # Path to your taichi repository
    export PYTHONPATH=$TAICHI_REPO_DIR/python/:$PYTHONPATH
    export PATH=$TAICHI_REPO_DIR/bin/:$PATH

- Execute ``source ~/.bashrc`` to reload shell config
- Execute ``ti test`` to run all the tests. It may take up to 5 minutes to run all tests.
- Check out ``examples`` for runnable examples. Run them with ``python3``.


Setting up CUDA 10.1 on Ubuntu 18.04
---------------------------------------------

First, make sure you have CUDA 10.1 installed.
Check this by running
``nvcc --version`` or ``cat /usr/local/cuda/version.txt``

If you don't have it - go ahead to `this website <https://developer.nvidia.com/cuda-downloads>`_ and download it.

These instructions were copied from the webiste above for x86_64 architecture

.. code-block:: bash

  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
  sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
  sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
  sudo apt-get update
  sudo apt-get -y install cuda

Prebuilt LLVM for Windows CI
-------------------------------------------------

.. code-block:: bash

  cmake .. -G"Visual Studio 15 2017 Win64"  -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64 -DLLVM_BUILD_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=installed

Then use Visual Studio to build. After building the ``INSTALL`` project (under folder "CMakePredefinedTargets"). After build completes, find your LLVM binaries/headers in `build/include`.

Folder structure
*************************************

Key folders are

- *analysis*: static analysis passes
- *backends*: codegen to x86 and CUDA
- *transforms*: IR transform passes
- ...

Troubleshooting
----------------------------------

- Run with debug mode to see if there's any illegal memory access
- Disable compiler optimizations to quickly confirm that the issue is not cause by optimization
