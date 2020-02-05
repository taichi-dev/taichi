Contribution guidelines (WIP)
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

Set ``ti.get_runtime().print_preprocessed = True``
to inspect results of the frontend Python AST transform. The resulting scripts will generate a Taichi Frontend AST when executed.

Set ``ti.cfg.print_ir = True`` to inspect the Taichi IR transformation process of kernel (excluding accessors) compilation .

Set ``ti.cfg.print_kernel_llvm_ir = True`` to inspect the emitted LLVM IR by Taichi.

Set ``ti.cfg.print_accessor_ir = True`` to inspect the IR transformation process of data accessors, which are special and simple kernels. (This is rarely used, unless you are debugging the compilation of data accessors.)

.. note::

  Data accessors in Python-scope are implemented as special Taichi kernels.
  For example, ``x[1, 2, 3] = 3`` will call the writing accessor kernel of ``x``,
  and ``print(y[42])`` will call the reading accessor kernel of ``y``.

Set ``ti.cfg.print_kernel_llvm_ir = True`` to inspect the emitted LLVM IR for each invoked kernel.

To navigate around the code base, [ffi-navigator](https://github.com/tqchen/ffi-navigator) allows you to jump from C++ extension symbols exported via taichi_core in Python code to their defintion in C++. Follow their README to set up your editor.

Testing
-------------

Tests should be added to ``taichi/tests/python``.

Use ``ti test`` to run all the tests.
(On Windows use, ``python -m taichi test``)

Documentation
-------------

Use ``ti doc`` to build the documentation locally.
Open the documentation at ``taichi/doc/build/index.html``.
