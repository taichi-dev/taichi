Installation
===============================================

Supported Platforms:
 - Ubuntu (gcc 5+)
 - Mac OS X (gcc 5+, clang 4.0+)
 - Windows (Microsoft Visual Studio 2017)

Make sure you have `python 3.5+`.


Ubuntu, Arch Linux, and Mac OS X
---------------------------------------

.. code-block:: bash

   wget https://raw.githubusercontent.com/yuanming-hu/taichi/legacy/install.py
   python3 install.py


Note, if python complains that a package is missing, simply rerun install.py and the package should be loaded.

Windows
-------------------------------
Download and execute `this script <https://raw.githubusercontent.com/yuanming-hu/taichi/master/install.py>`_ with python3.

Additional environment variables: (assuming taichi is installed in ``DIR/taichi``)
Set ``TAICHI_REPO_DIR`` as  ``DIR/taichi`` (e.g. ``E:/repos/taichi``).
Add ``%TAICHI_REPO_DIR%/python`` to ``PYTHONPATH``, ``DIR/taichi/bin`` (e.g. ``E:/repos/taichi/bin``) to ``PATH``.
Restart cmd or PowerShell, and you should be able to run command ``ti``.

Build with Double Precision (64 bit) Float Point
---------------------------------------------------
.. code-block:: bash

   export TC_USE_DOUBLE=1
   ti build

Examples
------------------------------
Please see `examples <https://github.com/yuanming-hu/taichi/tree/master/projects/examples>`_.

Run examples using `python3 [X.py]`. For example,

.. code-block:: bash

    python3 projects/examples/simulation/3d/mgpcg_smoke_3d.py
    python3 projects/examples/rendering/paper_cut.py

Please learn the python interface by examples for now.
Detailed documentation coming soon.
