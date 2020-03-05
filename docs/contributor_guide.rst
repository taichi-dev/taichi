Contribution guidelines
===============================================

First of all, thank you for contributing! We welcome contributions of
all forms, including but not limited to

- Bug fixes
- New features
- Documentation
- More user-friendly syntax error messages
- New example programs
- Compiler performance patches
- Minor typo fixes in the documentation, code, comments (please directly make a pull request for minor issues like these)

How to contribute bug fixes and new features
--------------------------------------------------

Issues marked with `"welcome contribution" <https://github.com/taichi-dev/taichi/issues?q=is%3Aopen+is%3Aissue+label%3A%22welcome+contribution%22>`_ are easy ones to start with.

- Please first leave a note (e.g. *I know how to fix this and would like to help!*) on the issue, so that people know some one is already working on it. ("Prevent redundant work";)

- If no lead developer has commented and described a potential solution on the issue, please also briefly describe your plan and wait for a lead developer to reply before you start. ("Keep solutions simple".)

High-level guidelines
---------------------

- Almost every design decision has pros and cons. Good decisions are good because their pros outweigh their cons. Always think of both sides of your decision.
- No overkills: always use the *easiest* solutions to solve easy problems, so that you have time and energy for real hard ones.
- Debugging is hard. Changesets should be small so that sources of bugs can be easily pinpointed.
- Unit/integration tests are our friends.

Commit rules
------------

- No commits with local (i.e., the contributor's local environment) compilation errors should be made;
- Commit messages should be concise and meaningful;
- The master branch is required to have a linear history.

Making good pull requests
-------------------------

- PRs with small changesets are preferred. A PR should ideally address **only one issue**.
- If you are making multiple PRs

 - Independent PRs should be based on **different** branches forking from ``master``;
 - PRs with dependencies should be raised only after all prerequisite PRs are merged into ``master``.

- All PRs should ideally come with corresponding **tests**;
- All PRs should come with **documentation update**, except for internal compiler implementations;
- All PRs should always be rebased onto the HEAD of master before merging;
- All PRs should pass continuous integration tests (build + testing for Mac/Windows) before they get merged;
- **PR authors should not squash commits**. Whether squashing all commits in a PR or not, will be decided by the reviewer who merges the PR into the master branch, depending on how trivially correct the commits are.


Reviewing & PR merging
----------------------

- Error-prone commits such as IR passes & codegen will be rebased on master (**without squashing**) once approved;
- Other commits with more trivial correctness (e.g. examples, GUI, benchmark cases, typo fixes) will first get squashed into a single commit and then rebased on master, for a cleaner master commit log.


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


Testing
-------

Tests should be added to ``taichi/tests/python``.

Use ``ti test`` to run all the tests. Use ``ti test_verbose`` to test with verbose outputs.



Documentation
-------------

Use ``ti doc`` to build the documentation locally.
Open the documentation at ``taichi/doc/build/index.html``.

C++ and Python standards
------------------------

The C++ part of Taichi is written in C++17, and Python part in 3.6+.
You can assume that C++17 and Python 3.6 features are always available.


(Linux only) pinpointing runtime errors using GDB
-------------------------------------------------
A quick way to pinpoint common runtime errors such as segmentation faults/assertion failures.
When Taichi crashes, ``gdb`` will be triggered and attach to the current thread.
You might be prompt to enter sudo password required for gdb thread attaching.
After entering ``gdb``, check the stack backtrace with command ``bt`` (``backtrace``),
then find the line of code triggering the error.


Efficient Code Navigation across Python/C++
-------------------------------------------
If you work on the language frontend (Python/C++ interface), to navigate around the code base, `ffi-navigator <https://github.com/tqchen/ffi-navigator>`_
allows you to jump from Python bindings to their definitions in C++.
Follow their README to set up your editor.


Folder structure
*************************************

Key folders are

- ``taichi``: The core compiler implementation

  - ``analysis``: Static analysis passes
  - ``runtime``: LLVMRuntime functions
  - ``backends``: Code generators
  - ``transforms``: IR transform passes
  - ``python_bindings``: C++/Python interfaces

- ``python``: Python frontend implementation
- ``examples``: Examples
- ``docs``: Documentation
- ``tests``: C++ and Python tests
- ``benchmarks``: Performance benchmarks
- ``misc``: Random (yet useful) files
- ...
