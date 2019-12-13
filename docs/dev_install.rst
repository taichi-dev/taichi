Developer Installation
=====================================================

Note this is for the compiler developers of Taichi lang. End users should use the pip packages instead of building from scratch.
Supports Ubuntu 14.04/16.04/18.04, ArchLinux, Mac OS X. For NVIDIA GPU support, CUDA 9.0+ is needed. 
See below for installing CUDA.

- Execute ``python3 -m pip install setuptools astpretty astor pytest opencv-python pybind11 Pillow numpy scipy GitPython yapf colorama psutil autograd``
- Execute ``sudo apt install libtinfo-dev clang-7`` on Ubuntu.
- Make sure you have LLVM 8.0.1 built from scratch, with (in the llvm source)

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
    cmake ..
    make -j 8

- Execute ``source ~/.bashrc`` (or ``source ~/.zshrc`` if you use ``zsh``) to reload shell config.
- (Optional, if you have NVIDIA GPU) Execute ``ti test`` to run all the tests. It may take a around 20 minutes to run all tests.
- Check out ``examples`` for runnable examples. Run them with `python3`.


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

Preparing Prebuilt LLVM for Windows CI
-------------------------------------------------

.. code-block:: bash

  cmake .. -G"Visual Studio 15 2017 Win64"  -DLLVM_ENABLE_RTTI:BOOL=ON -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_ASSERTIONS=ON -Thost=x64 -DLLVM_BUILD_TESTS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=installed

Then use Visual Studio to build. After building the "INSTALL" project, find your LLVM binaries/headers in `build/include`.

Folder Structure
*************************************

Key folders are
- *analysis*: static analysis passes
- *backends*: codegen to x86 and CUDA
- *transforms*: IR transform passes
- *ir*: the intermediate representation system
- *program*: the context for taichi programs
- ...

Troubleshooting
----------------------------------

- Run with debug mode to see if there's any illegal memory access; (TODO: this is broken in new releases)
- Disable compiler optimizations to quickly confirm that the issue is not cause by optimization;

Bibtex
------------------------------------

.. code-block:: python

  @inproceedings{hu2019taichi,
    title={Taichi: A Language for High-Performance Computation on Spatially Sparse Data Structures},
    author={Hu, Yuanming and Li, Tzu-Mao and Anderson, Luke and Ragan-Kelley, Jonathan and Durand, Fr\'edo},
    booktitle={SIGGRAPH Asia 2019 Technical Papers},
    pages={201},
    year={2019},
    organization={ACM}
  }