Contribution guidelines
===============================================

First of all, thank you for contributing! We welcome contributions of
all forms, including but not limited to

- Bug fixes
- New features
- Documentation
- More human-readable syntax error messages
- New example programs
- Compiler performance patches
- Minor typo fixes in the documentation, code, comments (please directly make a pull request for minor issues like these)

How to contribute bug fixes and new features
--------------------------------------------------

Issues marked with `"welcome contribution" <https://github.com/taichi-dev/taichi/issues?q=is%3Aopen+is%3Aissue+label%3A%22welcome+contribution%22>`_ are easy ones to start with.

- Please first leave a note (e.g. *I know how to fix this and would like to help!*) on the issue, so that people know some one is already working on it.

- If no lead developer has commented and described a potential solution on the issue, please also briefly describe your plan and wait for a lead developer to reply before you start.

Following these rules will prevent contributors from doing redundant work, and keep solutions simple and effective.

Tips on Taichi compiler development
--------------------------------------------------

:ref:`compilation` may worth checking out. It explains the whole compilation process.


When creating a Taichi program using ``ti.init(arch=desired_arch, **kwargs)``, pass in the following parameters to make the Taichi compiler print out IR:

    - ``print_preprocessed = True``: print results of the frontend Python AST transform. The resulting scripts will generate a Taichi Frontend AST when executed.
    - ``print_ir = True``: print the Taichi IR transformation process of kernel (excluding accessors) compilation.
    - ``print_kernel_llvm_ir = True``: print the emitted LLVM IR by Taichi.
    - ``print_kernel_llvm_ir_optimized = True``: print the optimized LLVM IR for each kernel.
    - ``print_accessor_ir = True``: print the IR transformation process of data accessors, which are special and simple kernels. (This is rarely used, unless you are debugging the compilation of data accessors.)

.. note::

  Data accessors in Python-scope are implemented as special Taichi kernels.
  For example, ``x[1, 2, 3] = 3`` will call the writing accessor kernel of ``x``,
  and ``print(y[42])`` will call the reading accessor kernel of ``y``.

Efficient Code Navigation across Python/C++
------------------------------------------------
If you work on the language frontend (Python/C++ interface), to navigate around the code base, `ffi-navigator <https://github.com/tqchen/ffi-navigator>`_
allows you to jump from Python bindings to their definitions in C++.
Follow their README to set up your editor.

Testing
-------------

Tests should be added to ``taichi/tests/python``.

Use ``ti test`` to run all the tests.
(On Windows, please use ``python -m taichi test``)

Documentation
-------------

Use ``ti doc`` to build the documentation locally.
Open the documentation at ``taichi/doc/build/index.html``.
