.. meta::
  :description: installing Taichi Lang for ROCm
  :keywords: installation instructions, parallel programming, JIT, LLVM, AMD, ROCm, Taichi Lang

.. _taichi-on-rocm-installation:

********************************************************************
Taichi Lang on ROCm installation
********************************************************************

System requirements
====================================================================

To use Taichi Lang `1.8.0b2 <https://github.com/ROCm/taichi/tree/release/1.8.0b2>`__, you need the following prerequisites:

- **ROCm version:** `7.0.0 <https://repo.radeon.com/rocm/apt/7.0/>`__ (recommended)
- **Operating system:** Ubuntu 22.04, 24.04
- **GPU platform:** AMD Instinct™ MI355X, MI325X, MI300X, MI250X, MI210
- **Python:** `3.12.3 <https://www.python.org/downloads/release/python-3123/>`__, `3.10.12 <https://www.python.org/downloads/release/python-31012/>`__

Install Taichi
================================================================================

To install Taichi Lang on ROCm, you have the following options:

- :ref:`Use the prebuilt Docker image <using-docker-with-taichi-pre-installed>` **(recommended)**
- :ref:`Use a wheels package <taichi-wheels-package>`
- :ref:`Build your own docker image <build-taichi-rocm-docker-image>`

.. _using-docker-with-taichi-pre-installed:

Use a prebuilt Docker image with Taichi Lang pre-installed
--------------------------------------------------------------------------------

Docker is the recommended method to set up a Taichi Lang environment, as it avoids potential installation issues. 
The tested, prebuilt image includes Taichi, Python, ROCm, and other dependencies.

1. Pull the Docker image:

   .. tab-set::

      .. tab-item:: Ubuntu 24.04
         :sync: ubuntu-24

         .. code-block:: shell

            docker pull rocm/taichi:taichi-1.8.0b2_rocm7.0.0_ubuntu24.04_py3.12.3

         See `rocm/taichi:taichi-1.8.0b2_rocm7.0.0_ubuntu24.04_py3.12.3
         <https://hub.docker.com/layers/rocm/taichi/taichi-1.8.0b2_rocm7.0.0_ubuntu24.04_py3.12.3/images/sha256-15bb5ad69e0dff0cc5e0805a5d271a82d04d6d9c32d668bfcf34981ca99f68eb>`__
         on Docker Hub.

      .. tab-item:: Ubuntu 22.04
         :sync: ubuntu-22

         .. code-block:: shell

            docker pull rocm/taichi:taichi-1.8.0b2_rocm7.0.0_ubuntu22.04_py3.10.12

         See `rocm/taichi:taichi-1.8.0b2_rocm7.0.0_ubuntu22.04_py3.10.12
         <https://hub.docker.com/layers/rocm/taichi/taichi-1.8.0b2_rocm7.0.0_ubuntu22.04_py3.10.12/images/sha256-151516ac981ab1309fca0d868e25dd1231728449b6563b5e8c660d7cb47777fb>`__
         on Docker Hub.

2. Launch and connect to the container:

   .. tab-set::

      .. tab-item:: Ubuntu 24.04
         :sync: ubuntu-24

         .. code-block:: shell

            docker run -it -d \
               --cap-add=SYS_PTRACE \
               --security-opt seccomp=unconfined \
               --ipc=host \
               --shm-size=64G \
               --network=host \
               --device=/dev/kfd \
               --device=/dev/dri \
               --group-add video \
               -v "$(pwd)":/taichi_dir \
               --name rocm_taichi \
               rocm/taichi:taichi-1.8.0b2_rocm7.0.0_ubuntu24.04_py3.12.3

      .. tab-item:: Ubuntu 22.04
         :sync: ubuntu-22

         .. code-block:: shell

            docker run -it -d \
               --cap-add=SYS_PTRACE \
               --security-opt seccomp=unconfined \
               --ipc=host \
               --shm-size=64G \
               --network=host \
               --device=/dev/kfd \
               --device=/dev/dri \
               --group-add video \
               -v "$(pwd)":/taichi_dir \
               --name rocm_taichi \
               rocm/taichi:taichi-1.8.0b2_rocm7.0.0_ubuntu22.04_py3.10.12

.. _taichi-wheels-package:

Use a wheels package
--------------------------------------------------------------------------------

The Taichi Lang ``.whl`` packages are hosted on the AMD PyPI repository. 
Instead of manually downloading the files, you can simply install Taichi Lang using ``pip`` with the provided URL. 
This command will automatically download and install the appropriate ``.whl`` file.

.. code-block:: bash

   pip install amd-taichi==1.8.0b2 --index-url=https://pypi.amd.com/simple
   sudo apt-get update
   sudo apt-get install -y lld

.. _build-taichi-rocm-docker-image:

Build your own Docker image
--------------------------------------------------------------------------------

If you prefer to use the ROCm Ubuntu image, or already have a ROCm Ubuntu container, follow these steps to install Taichi in the container.

1. Pull the ROCm Ubuntu Docker image:

   .. tab-set::

      .. tab-item:: Ubuntu 24.04
         :sync: ubuntu-24

         .. code-block:: shell

            docker pull rocm/dev-ubuntu-24.04:7.0-complete

         See `rocm/dev-ubuntu-24.04:7.0-complete
         <https://hub.docker.com/layers/rocm/dev-ubuntu-24.04/7.0-complete/images/sha256-ffd8ac00ca6c8e2dbfd0c364c7cc27542f90148f3f358d74efd028f67c33607b>`__
         on Docker Hub.

      .. tab-item:: Ubuntu 22.04
         :sync: ubuntu-22

         .. code-block:: shell

            docker pull rocm/dev-ubuntu-22.04:7.0-complete

         See `rocm/dev-ubuntu-22.04:7.0-complete
         <https://hub.docker.com/layers/rocm/dev-ubuntu-22.04/7.0-complete/images/sha256-b4be4b0b29e46d56e9bea2cd06500f4519aaac30dc5df02bd4710bbf393c1c4c>`__
         on Docker Hub.

2. Launch the Docker container:

   .. tab-set::

      .. tab-item:: Ubuntu 24.04
         :sync: ubuntu-24

         .. code-block:: shell

            docker run -it -d \
                  --cap-add=SYS_PTRACE \
                  --security-opt seccomp=unconfined \
                  --ipc=host \
                  --shm-size=64G \
                  --network=host \
                  --device=/dev/kfd \
                  --device=/dev/dri \
                  --group-add video \
                  -v "$(pwd)":/taichi_dir \
                  --name rocm_taichi \
                  rocm/dev-ubuntu-24.04:7.0-complete \
                  /bin/bash

      .. tab-item:: Ubuntu 22.04
         :sync: ubuntu-22

         .. code-block:: shell

            docker run -it -d \
                  --cap-add=SYS_PTRACE \
                  --security-opt seccomp=unconfined \
                  --ipc=host \
                  --shm-size=64G \
                  --network=host \
                  --device=/dev/kfd \
                  --device=/dev/dri \
                  --group-add video \
                  -v "$(pwd)":/taichi_dir \
                  --name rocm_taichi \
                  rocm/dev-ubuntu-22.04:7.0-complete \
                  /bin/bash

3. Inside the running container, install build dependencies:

   .. code-block:: bash

      apt-get update && apt-get install -y --no-install-recommends \
         git wget vim git freeglut3-dev libglfw3-dev libglm-dev \
         libglu1-mesa-dev libjpeg-dev liblz4-dev libpng-dev \
         libssl-dev libwayland-dev libx11-xcb-dev libxcb-dri3-dev \
         libxcb-ewmh-dev libxcb-keysyms1-dev libxcb-randr0-dev \
         libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev \
         libzstd-dev python3-pip cmake pybind11-dev \
         ca-certificates python3-venv rocm-llvm-dev \
         gdb python3-dbg 

4. Setup LLVM:

   .. code-block:: bash
      
      export LLVM_VERSION=20
      export LLVM_PATH=/usr/lib/llvm-${LLVM_VERSION}
      export PATH=${LLVM_PATH}/bin:$PATH
      
      wget https://apt.llvm.org/llvm.sh \
         && chmod +x llvm.sh \
         && apt-get update && apt-get install -y \
         lsb-release software-properties-common gnupg \
         && ./llvm.sh ${LLVM_VERSION} llvm clang lld

5. Clone the `https://github.com/ROCm/taichi <https://github.com/ROCm/taichi>`_ repository with the desired branch:

   .. code-block:: bash
      
      cd /taichi_dir
      git clone --recursive https://github.com/ROCm/taichi -b release/v1.8.0b2
      cd taichi

6. Build the Taichi Lang wheel:

   .. code-block:: bash

      export GPU_TARGETS=gfx950,gfx942,gfx90a
      export TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN=OFF -DTI_WITH_OPENGL=OFF -DTI_BUILD_TESTS=ON -DTI_BUILD_EXAMPLES=OFF -DCMAKE_PREFIX_PATH=${LLVM_PATH}/lib/cmake -DCMAKE_CXX_COMPILER=${LLVM_PATH}/bin/clang++ -DTI_WITH_AMDGPU=ON -DTI_WITH_CUDA=OFF -DTI_AMDGPU_ARCHS=${GPU_TARGETS} -DUSE_LLD=ON -DTI_WITH_LLVM=ON"

      cd /taichi_dir/taichi/external/spdlog \
      && git apply /taichi_dir/taichi/spdlog_fmt.patch \
      && cd /taichi_dir/taichi \
      && ./build.py

7. Install the Taichi Lang ``.whl`` file:

   .. code-block:: bash

      python3 -m pip config set global.break-system-packages true \
      && python3 -m pip install /taichi_dir/taichi/dist/taichi*.whl


.. _build-taichi-docker-from-source:

Test the Taichi Lang installation
================================================================================

Clone the `https://github.com/ROCm/taichi <https://github.com/ROCm/taichi>`_ repository:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y git
   git clone --recursive https://github.com/ROCm/taichi -b amd-release/v1.8.0b2

To test the Taichi Lang installation, run the ``laplace`` example in the source code: 

.. code-block:: bash

   python3 taichi/python/taichi/examples/algorithm/laplace.py

Example output using ``laplace``:

.. code-block:: bash

   [Taichi] version 1.8.0, llvm 15.0.0, commit f7911653, linux, python python 3.12.3
   [Taichi] Starting on arch=amdgpu
   0.0
   4.0
   0.0
   0.0
   4.0
   0.0
   0.0
   4.0
   0.0
   0.0

Run a Taichi Lang example
====================================================================

A set of examples is available to help you get started. See :doc:`run a Taichi Lang example <../examples/taichi-examples>` for more details.
