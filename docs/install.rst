Installation
============

First of all, let's install Taichi via ``pip``:

.. code-block:: bash

  python3 -m pip install taichi

.. note::

    Currently, Taichi only support Python 3.6/3.7/3.8, versions below are **not supported**.


- On Ubuntu 19.04+, please execute ``sudo apt install libtinfo5``.
- On Arch Linux, please execute ``yaourt -S ncurses5-compat-libs``.
- On Windows, please install `Microsoft Visual C++ Redistributable <https://aka.ms/vs/16/release/vc_redist.x64.exe>`_ if you haven't.


Troubleshooting
---------------

If you found Taichi crashed with following message:

.. code-block::

    [Taichi] mode=release
    [Taichi] version 0.6.0, supported archs: [cpu, cuda, opengl], commit 14094f25, python 3.8.2
    [W 05/14/20 10:46:49.549] [cuda_driver.h:call_with_warning@60] CUDA Error CUDA_ERROR_INVALID_DEVICE: invalid device ordinal while calling mem_advise (cuMemAdvise)
    [E 05/14/20 10:46:49.911] Received signal 7 (Bus error)


This may because your NVIDIA card is pre-Pascal and therefore does not support `Unified Memory <https://www.nextplatform.com/2019/01/24/unified-memory-the-final-piece-of-the-gpu-programming-puzzle/>`_.

* Try adding ``export TI_USE_UNIFIED_MEMORY=0`` to your ``~/.bashrc``, this disables CUDA backend to use unified memory support.


If you found other problems related to CUDA:

* Try adding ``export TI_ENABLE_CUDA=0`` to your  ``~/.bashrc``, this disables CUDA backend completely and Taichi will fall back on other GPU backends like OpenGL.


If you found Taichi crashed with Stack Traceback containing a line of ``glfwCreateWindow`` (see `#958 <https://github.com/taichi-dev/taichi/issues/958>`_):

.. code-block::

    [Taichi] mode=release
    [E 05/12/20 18.25:00.129] Received signal 11 (Segmentation Fault)
    ***********************************
    * Taichi Compiler Stack Traceback *
    ***********************************

    ... (many lines, omitted)

    /lib/python3.8/site-packages/taichi/core/../lib/taichi_core.so: _glfwPlatformCreateWindow
    /lib/python3.8/site-packages/taichi/core/../lib/taichi_core.so: glfwCreateWindow
    /lib/python3.8/site-packages/taichi/core/../lib/taichi_core.so: taichi::lang::opengl::initialize_opengl(bool)

    ... (many lines, omitted)

This may because you are running Taichi on VM which has a very low OpenGL version (while we requires OpenGL 4.3+ to work):

* Try adding ``export TI_ENABLE_OPENGL=0`` to your  ``~/.bashrc``, this disables OpenGL backend and Taichi will fall back on CPU backend.


If you found Taichi crashed with messages reporting ``libtinfo.so.5`` missing:

* Please install ``libtinfo5`` for Ubuntu or ``ncurses5-compat-libs`` (AUR) for Arch Linux.
