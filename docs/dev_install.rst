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

    python3 -m pip install --user setuptools wheel astor pybind11 Pillow
    python3 -m pip install --user pytest pytest-rerunfailures pytest-xdist yapf
    python3 -m pip install --user numpy GitPython coverage colorama autograd


- Make sure you have ``clang`` with version >= 7

  * On Windows: Download ``clang-8`` via `this link <https://releases.llvm.org/8.0.0/LLVM-8.0.0-win64.exe>`_.
    Make sure you add the ``bin`` folder containing ``clang.exe`` to the ``PATH`` environment variable.

  * On OS X: you don't need to do anything.

  * On Ubuntu, execute ``sudo apt install clang-8``.
  
  * On Arch Linux, execute ``sudo pacman -S clang``. (this will install ``clang-10``, which should work as well).

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

- Add the following script to your ``~/.bashrc``:

  .. code-block:: bash

    export TAICHI_REPO_DIR=/home/XXX/taichi  # Path to your taichi repository
    export PYTHONPATH=$TAICHI_REPO_DIR/python:$PYTHONPATH
    export PATH=$TAICHI_REPO_DIR/bin:$PATH
    # export PATH=/opt/llvm/bin:$PATH # Uncomment if your llvm-8 or clang-8 is in /opt
    export CMAKE_CXX_COMPILER=clang++

- Execute ``source ~/.bashrc`` to reload shell config.

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

- Execute ``python3 -m taichi test`` to run all the tests. It may take up to 5 minutes to run all tests.
- Check out ``examples`` for runnable examples. Run them with ``python3``.


Developer's Troubleshooting
---------------------------

- If ``make`` failed to compile and reporting ``fatal error: 'spdlog/XXX.h' file not found``,
  please try run ``git submodule init --update --recursive --depth=1``.

- If ``cmake`` failed to generate and reporting ``clang and MSVC are the only supported compilers for Taichi compiler development``,
  please make sure you have installed clang with version >= 7, and add ``export CMAKE_CXX_COMPILER=/path/to/clang`` to ``.bashrc``.

- If importing Taichi cause ``FileNotFoundError: [Errno 2] No such file or directory: '/root/taichi/python/taichi/core/../lib/taichi_core.so' -> '/root/taichi/python/taichi/core/../lib/libtaichi_core.so'``, please:

  * On Windows, add a environment variable called ``TAICHI_REPO_DIR`` with value ``/home/XXX/taichi`` (Taichi repo path), this may be either done in your IDE configuation panel or system settings.
  * On Linux / OS X, add ``export TAICHI_REPO_DIR=/home/XXX/taichi`` to your ``~/.bashrc`` and restart ``bash``.

- If the build succeed, but running a Taichi example result in an error, please make sure ``clang`` is in your ``PATH``:

  .. code-block:: bash
  
      clang --version
      # version should be >= 7
      
  and our **Taichi configured** ``llvm-as``:

  .. code-block:: bash
  
      llvm-as --version
      # version should be 8.0.1 or 10.0.0
      which llvm-as
      # should be /usr/local/bin/llvm-as or /opt/XXX/bin/llvm-as, which is our configured installation
      
  If not, please install ``clang`` and **build LLVM from source** with instructions above in `:ref:dev_install`, then add their path to environment variable ``PATH``.

- If you encountered other issues, feel free to report it by `opening an issue on GitHub <https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md>`_, we are willing to help!

- See also `:ref:troubleshooting` for issues that may share with end-user installation.
