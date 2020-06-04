Installation
============

Taichi can be easily installed via ``pip``:

.. code-block:: bash

  python3 -m pip install taichi

.. note::

    Currently, Taichi only supports Python 3.6/3.7/3.8 (64-bit).


- On Ubuntu 19.04+, please execute ``sudo apt install libtinfo5``.
- On Arch Linux, please execute ``yaourt -S ncurses5-compat-libs``.
- On Windows, please install `Microsoft Visual C++ Redistributable <https://aka.ms/vs/16/release/vc_redist.x64.exe>`_ if you haven't.


Troubleshooting
---------------

Windows issues
**************

- If Taichi crashes and reports ``ImportError`` on Windows: Please consider installing `Microsoft Visual C++ Redistributable <https://aka.ms/vs/16/release/vc_redist.x64.exe>`_.

Python issues
*************

- If ``pip`` complains that it could not find a satisfying package, i.e.,

  .. code-block:: none

    ERROR: Could not find a version that satisfies the requirement taichi (from versions: none)
    ERROR: No matching distribution found for taichi

  * Make sure you're using Python version 3.6/3.7/3.8:

    .. code-block:: bash

      python3 -c "print(__import__('sys').version[:3])"
      # 3.6, 3.7 or 3.8

  * Make sure your Python executable is 64-bit:

    .. code-block:: bash

      python3 -c "print(__import__('platform').architecture()[0])"
      # 64bit

CUDA issues
***********

- If Taichi crashes with the following messages:

    .. code-block:: none

        [Taichi] mode=release
        [Taichi] version 0.6.0, supported archs: [cpu, cuda, opengl], commit 14094f25, python 3.8.2
        [W 05/14/20 10:46:49.549] [cuda_driver.h:call_with_warning@60] CUDA Error CUDA_ERROR_INVALID_DEVICE: invalid device ordinal while calling mem_advise (cuMemAdvise)
        [E 05/14/20 10:46:49.911] Received signal 7 (Bus error)


  This might be due to the fact that your NVIDIA GPU is pre-Pascal and has limited support for `Unified Memory <https://www.nextplatform.com/2019/01/24/unified-memory-the-final-piece-of-the-gpu-programming-puzzle/>`_.

  * **Possible solution**: add ``export TI_USE_UNIFIED_MEMORY=0`` to your ``~/.bashrc``. This disables unified memory usage in CUDA backend.


- If you find other CUDA problems:

  * **Possible solution**: add ``export TI_ENABLE_CUDA=0`` to your  ``~/.bashrc``. This disables the CUDA backend completely and Taichi will fall back on other GPU backends such as OpenGL.

OpenGL issues
*************

- If Taichi crashes with a stack backtrace containing a line of ``glfwCreateWindow`` (see `#958 <https://github.com/taichi-dev/taichi/issues/958>`_):

  .. code-block:: none

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

  This is likely because you are running Taichi on a (virtual) machine with an old OpenGL API. Taichi requires OpenGL 4.3+ to work.

  * **Possible solution**: add ``export TI_ENABLE_OPENGL=0`` to your  ``~/.bashrc`` even if you initialize Taichi with other backends than OpenGL. This disables the OpenGL backend detection to avoid incompatibilities.


Linux issues
************

- If Taichi crashes and reports ``libtinfo.so.5 not found``: Please install ``libtinfo5`` for Ubuntu or ``ncurses5-compat-libs`` (AUR) for Arch Linux.


Other issues
************

- If none of those above address your problem, please report this by `opening an issue <https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md>`_ on GitHub. This would help us improve user experiences and compatibility, many thanks!
