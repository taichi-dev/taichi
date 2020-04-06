Contribution guidelines
=======================

First of all, thank you for contributing! We welcome contributions of
all forms, including but not limited to

- Bug fixes
- New features
- Documentation
- Improved error messages that are more user-friendly
- New example programs
- Compiler performance patches
- Minor typo fixes in the documentation, code, comments (please directly make a pull request for minor issues like these)

How to contribute bug fixes and new features
--------------------------------------------

Issues marked with `"welcome contribution" <https://github.com/taichi-dev/taichi/issues?q=is%3Aopen+is%3Aissue+label%3A%22welcome+contribution%22>`_ are easy ones to start with.

- Please first leave a note (e.g. *I know how to fix this and would like to help!*) on the issue, so that people know someone is already working on it. This helps prevent redundant work;

- If no core developer has commented and described a potential solution on the issue, please briefly describe your plan, and wait for a core developer to reply before you start.
  This helps keep implementations simple and effective.

High-level guidelines
---------------------

- Be pragmatic: practically solving problems is our ultimate goal.
- No overkills: always use *easy* solutions to solve easy problems, so that you have time and energy for real hard ones.
- Almost every design decision has pros and cons. A decision is `good` if its pros outweigh its cons. Always think about both sides.
- Debugging is hard. Changesets should be small so that sources of bugs can be easily pinpointed.
- Unit/integration tests are our friends.

.. note::
  “There are two ways of constructing a software design: One way is to make it so simple that there are obviously no deficiencies, and the other way is to make it so complicated that there are no obvious deficiencies. `The first method is far more difficult`.”     --- `C.A.R. Hoare <https://en.wikipedia.org/wiki/Tony_Hoare>`_

Effective communication
-----------------------

- How much information we effectively convey, is way more important than how many words we typed.
- Be constructive. Be polite. Be organized. Be concise.
- Bulleted lists are our friends.
- Proofread before you post: if you are the reader, can you understand what you typed?
- If you are not a native speaker, consider using a spell checker such as `Grammarly <https://app.grammarly.com/>`_.


Making good pull requests
-------------------------

- PRs with **small** changesets are preferred. A PR should ideally address **only one issue**.
- All commits in a PR will always be **squashed and merged into master as a single commit**.
- When implementing a complex feature, consider breaking it down into small PRs, to keep a more detailed development history and to interact with core developers more frequently.
- If you want early feedback from core developers, open a PR in **Draft** state on GitHub so that you can share your progress.
- If you are making multiple PRs

  - Independent PRs should be based on **different** branches forking from ``master``;
  - PRs with dependencies should be raised only after all prerequisite PRs are merged into ``master``.

- All PRs should ideally come with corresponding **tests**;
- All PRs should come with **documentation update**, except for internal compiler implementations;
- All PRs should always be **rebased** onto the latest master branch before merging;
- All PRs should pass **continuous integration tests** before they get merged;
- PR authors **should not squash commits on their own**;
- PR titles should follow :ref:`prtag`;


Reviewing & PR merging
----------------------

- The merger should always **squash and merge** PRs into the master branch;
- The master branch is required to have a **linear history**;
- Make sure the PR passes **continuous integration tests**, except for cases like documentation updates;
- Make sure the title follows :ref:`prtag`.


.. _prtag:

PR title tags
-------------
Please always prepend exactly one tag such as ``[Metal]`` to PR titles. For example, "[Metal] Support bitmasked SNode", "[OpenGL] AtomicMin/Max support", or "[Opt] Enhanced constant folding".

Existing tags:

- ``[Metal], [OpenGL], [CPU], [CUDA], [AMDGPU], [LLVM]``: backends;
- ``[Lang]``: frontend language features, including syntax sugars;
- ``[Std]``: standard library, e.g. `ti.Matrix` and `ti.Vector`;
- ``[IR]``: intermediate representation;
- ``[Opt]``: IR optimization passes;
- ``[Async]``: asynchronous execution engine;
- ``[Type]``: type system;
- ``[Infra]``: general infrastructure, e.g. logging, image reader;
- ``[GUI]``: the built-in GUI system;
- ``[Refactor]``: code refactoring;
- ``[AutoDiff]``: automatic differentiation;
- ``[CLI]``: commandline interfaces, e.g. the ``ti`` command;
- ``[Doc]``: documentation;
- ``[Example]``: examples under ``taichi/examples/``;
- ``[Test]``: adding or improving tests under ``tests/``;
- ``[Misc]``: something that doesn't belong to any category, such as version bump, reformatting;
- **When introducing a new tag, please update the list here in the first PR with that tag, so that people can follow.**

.. note::

  We do appreciate all kinds of contributions, yet we should not expose the title of every PR to end-users.
  Therefore the changelog will distinguish `what the user should know` from `what the developers are doing`.
  This is done by **capitalizing PR tags**:

   - PRs with visible/notable features to the users should be marked with tags starting with **the first letter capitalized**, e.g. ``[Metal], [OpenGL], [IR], [Lang], [CLI]``.
     When releasing a new version, a script will generate a changelog with these changes (PR title) highlighted. Therefore it is **important** to make sure the end-users can understand what your PR does, **based on your PR title**.
   - Other PRs (underlying development/intermediate implementation) should use tags with **everything in lowercase letters**: e.g. ``[metal], [opengl], [ir], [lang], [cli]``.

Tips on the Taichi compiler development
---------------------------------------

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

Tests should be added to ``taichi/tests``.
Use ``ti test`` to run all the tests.
Use ``ti test -v`` for verbose outputs.
Use ``ti test <filename(s)>`` to run specific tests. Feel free to omit the ``test_`` prefix and ``.py`` suffix in the filenames.

For more options, see ``ti test -h``.

Documentation
-------------

Use ``ti doc`` to build the documentation locally.
Open the documentation at ``taichi/doc/build/index.html``.
On Linux/OS X, use ``watch -n 1 ti doc`` to continuously build the documentation.

C++ and Python standards
------------------------

The C++ part of Taichi is written in C++17, and the Python part in 3.6+.
You can assume that C++17 and Python 3.6 features are always available.


(Linux only) pinpointing runtime errors using ``gdb``
-----------------------------------------------------
A quick way to pinpoint common runtime errors such as segmentation faults/assertion failures.
When Taichi crashes, ``gdb`` will be triggered and attach to the current thread.
You might be prompt to enter sudo password required for gdb thread attaching.
After entering ``gdb``, check the stack backtrace with command ``bt`` (``backtrace``),
then find the line of code triggering the error.


Efficient code navigation across Python/C++
-------------------------------------------
If you work on the language frontend (Python/C++ interface), to navigate around the code base, `ffi-navigator <https://github.com/tqchen/ffi-navigator>`_
allows you to jump from Python bindings to their definitions in C++.
Follow their README to set up your editor.


Folder structure
----------------

Key folders are

- ``taichi``: The core compiler implementation

  - ``program``: Top-level constructs
  - ``runtime``: Runtime environments
  - ``codegen``: Code generators
  - ``struct``: Struct compilers
  - ``backends``: Device-dependent code generators/runtime environments
  - ``llvm``: LLVM utils
  - ``ir``: Intermediate representation
  - ``transforms``: IR transform passes
  - ``analysis``: Static analysis passes
  - ``python``: C++/Python interfaces

- ``python``: Python frontend implementation
- ``examples``: Examples
- ``docs``: Documentation
- ``tests``: C++ and Python tests
- ``benchmarks``: Performance benchmarks
- ``misc``: Random (yet useful) files
- ...
