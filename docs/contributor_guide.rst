Contribution guidelines
=======================

First of all, thank you for contributing! We welcome contributions of
all forms, including but not limited to

- Bug fixes
- Proposing and implementing new features
- Documentation improvement and translations (e.g. `Simplified Chinese <https://github.com/taichi-dev/taichi-docs-zh-cn>`_)
- Improved error messages that are more user-friendly
- New test cases
- New examples
- Compiler performance patches
- Blog posts and tutorials on Taichi
- Participation in the `Taichi forum <https://forum.taichi.graphics/>`_
- Introduce Taichi to your friends or simply star `the project <https://github.com/taichi-dev/taichi>`_.
- Typo fixes in the documentation, code or comments (please directly make a pull request for minor issues like these)

How to contribute bug fixes and new features
--------------------------------------------

Issues marked with `"good first issue" <https://github.com/taichi-dev/taichi/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_ are great chances for starters.

- Please first leave a note (e.g. *I know how to fix this and would like to help!*) on the issue, so that people know someone is already working on it. This helps prevent redundant work;

- If no core developer has commented and described a potential solution on the issue, please briefly describe your plan, and wait for a core developer to reply before you start.
  This helps keep implementations simple and effective.

Issues marked with `"welcome contribution" <https://github.com/taichi-dev/taichi/issues?q=is%3Aopen+is%3Aissue+label%3A%22welcome+contribution%22>`_ are slightly more challenging but still friendly to beginners.

High-level guidelines
---------------------

- Be pragmatic: practically solving problems is our ultimate goal.
- No overkills: always use *easy* solutions to solve easy problems, so that you have time and energy for real hard ones.
- Almost every design decision has pros and cons. A decision is `good` if its pros outweigh its cons. Always think about both sides.
- Debugging is hard. Changesets should be small so that sources of bugs can be easily pinpointed.
- Unit/integration tests are our friends.

.. note::
  “There are two ways of constructing a software design: One way is to make it so simple that there are obviously no deficiencies, and the other way is to make it so complicated that there are no obvious deficiencies. `The first method is far more difficult`.”     --- `C.A.R. Hoare <https://en.wikipedia.org/wiki/Tony_Hoare>`_

One thing to keep in mind is that, Taichi was originally born as an academic research project. This usually means that some parts did not have the luxury to go through a solid design. While we are always trying to improve the code quality, it doesn't mean that the project is free from technical debts. Some places may be confusing or overly complicated. Whenever you spot one, you are more than welcome to shoot us a PR! :-)

Effective communication
-----------------------

- How much information we effectively convey, is way more important than how many words we typed.
- Be constructive. Be polite. Be organized. Be concise.
- Bulleted lists are our friends.
- Proofread before you post: if you are the reader, can you understand what you typed?
- If you are not a native speaker, consider using a spell checker such as `Grammarly <https://app.grammarly.com/>`_.

Please base your discussion and feedback on facts, and not personal feelings. It is very important for all of us to maintain a friendly and blame-free community. Some examples:

.. tip::
  (Acceptable) This design could be confusing to new Taichi users.

.. warning::
  (Not Acceptable) This design is terrible.


Making good pull requests
-------------------------

- PRs with **small** changesets are preferred. A PR should ideally address **only one issue**.

  - It is fine to include off-topic **trivial** refactoring such as typo fixes;
  - The reviewers reserve the right to ask PR authors to remove off-topic **non-trivial** changes.

- All commits in a PR will always be **squashed and merged into master as a single commit**.
- PR authors **should not squash commits on their own**;
- When implementing a complex feature, consider breaking it down into small PRs, to keep a more detailed development history and to interact with core developers more frequently.
- If you want early feedback from core developers

  - Open a PR in `Draft <https://github.blog/2019-02-14-introducing-draft-pull-requests/>`_ state on GitHub so that you can share your progress;
  - Make sure you @ the corresponding developer in the comments or request the review.

- If you are making multiple PRs

  - Independent PRs should be based on **different** branches forking from ``master``;
  - PRs with dependencies should be raised only after all prerequisite PRs are merged into ``master``.

- All PRs should ideally come with corresponding **tests**;
- All PRs should come with **documentation update**, except for internal compiler implementations;
- All PRs must pass **continuous integration tests** before they get merged;
- PR titles should follow :ref:`prtag`;
- A great article from Google on `how to have your PR merged quickly <https://testing.googleblog.com/2017/06/code-health-too-many-comments-on-your.html>`_. `[PDF] <https://github.com/yuanming-hu/public_files/blob/master/graphics/taichi/google_review_comments.pdf>`_


Reviewing & PR merging
----------------------

- Please try to follow these tips from Google

  - `Code Health: Understanding Code In Review <https://testing.googleblog.com/2018/05/code-health-understanding-code-in-review.html>`_; `[PDF] <https://github.com/yuanming-hu/public_files/blob/master/graphics/taichi/google_understanding_code.pdf>`_
  - `Code Health: Respectful Reviews == Useful Reviews <https://testing.googleblog.com/2019/11/code-health-respectful-reviews-useful.html>`_. `[PDF] <https://github.com/yuanming-hu/public_files/blob/master/graphics/taichi/google_respectful_reviews.pdf>`_

- The merger should always **squash and merge** PRs into the master branch;
- The master branch is required to have a **linear history**;
- Make sure the PR passes **continuous integration tests**, except for cases like documentation updates;
- Make sure the title follows :ref:`prtag`.


Using continuous integration
----------------------------

- Continuous Integration (CI), will **build** and **test** your commits in a PR against in environments.
- Currently, Taichi uses `Travis CI <https://travis-ci.org>`_ (for OS X and Linux) and `AppVeyor <https://www.appveyor.com>`_ (for Windows).
- CI will be triggered every time you push commits to an open PR.
- You can prepend ``[skip ci]`` to your commit message to avoid triggering CI. e.g. ``[skip ci] This commit will not trigger CI``
- A tick on the right of commit hash means CI passed, a cross means CI failed.


Enforcing code style
--------------------
- Locally, you can run ``ti format`` in the command line to re-format code style.
  Note that you have to install ``clang-format-6.0`` and ``yapf v0.29.0`` locally before you use ``ti format``.

- If you don't have to install these formatting tools locally, use the **format server**. It's an online version of ``ti format``.

   - Go to http://kun.csail.mit.edu:31415/, and click at the desired PR id.
   - Come back to the PR page, you'll see a user called @taichi-gardener (bot) pushed a commit named ``[skip ci] enforce code format``.
   - You won't see the bot's commit if it didn't find anything not matching the format.
   - Then please run ``git pull`` in your local branch to pull the formatted code.
   - Note that commit messages marked with ``[format]`` will automatically trigger the format server. e.g. ``[format] your commit message``


.. _prtag:

PR title format and tags
------------------------
PR titles will be part of the commit history reflected in the ``master`` branch, therefore it is important to keep PR titles readable.

 - Please always prepend **at least one tag** such as ``[Lang]`` to PR titles:

     - When using multiple tags, make sure there is exactly one space between tags;
     - E.g., "[Lang][refactor]" (no space) should be replaced by "[Lang] [refactor]";

 - The first letter of the PR title body should be capitalized:

     - E.g., ``[Doc] improve documentation`` should be replaced by ``[Doc] Improve documentation``;
     - ``[Lang] "ti.sqr(x)" is now deprecated`` is fine because ``"`` is a symbol.

 - Please do not include back quotes ("`") in PR titles.

 - For example, "[Metal] Support bitmasked SNode", "[OpenGL] AtomicMin/Max support", or "[Opt] [IR] Enhanced constant folding".

Frequently used tags:

- ``[Metal], [OpenGL], [CPU], [CUDA]``: backends;
- ``[LLVM]``: the LLVM backend shared by CPUs and CUDA;
- ``[Lang]``: frontend language features, including syntax sugars;
- ``[Std]``: standard library, e.g. ``ti.Matrix`` and ``ti.Vector``;
- ``[Sparse]``: sparse computation;
- ``[IR]``: intermediate representation;
- ``[Opt]``: IR optimization passes;
- ``[GUI]``: the built-in GUI system;
- ``[Refactor]``: code refactoring;
- ``[CLI]``: commandline interfaces, e.g. the ``ti`` command;
- ``[Doc]``: documentation under ``docs/``;
- ``[Example]``: examples under ``examples/``;
- ``[Test]``: adding or improving tests under ``tests/``;
- ``[Linux]``: Linux platform;
- ``[Mac]``: Mac OS X platform;
- ``[Windows]``: Windows platform;
- ``[Perf]``: performance improvements;
- ``[Misc]``: something that doesn't belong to any category, such as version bump, reformatting;
- ``[Bug]``: bug fixes;
- Check out more tags in `misc/prtags.json <https://github.com/taichi-dev/taichi/blob/master/misc/prtags.json>`_.
- When introducing a new tag, please update the list in ``misc/prtags.json`` in the first PR with that tag, so that people can follow.

.. note::

  We do appreciate all kinds of contributions, yet we should not expose the title of every PR to end-users.
  Therefore the changelog will distinguish `what the user should know` from `what the developers are doing`.
  This is done by **capitalizing PR tags**:

   - PRs with visible/notable features to the users should be marked with tags starting with **the first letter capitalized**, e.g. ``[Metal], [OpenGL], [IR], [Lang], [CLI]``.
     When releasing a new version, a script (``python/taichi/make_changelog.py``) will generate a changelog with these changes (PR title) highlighted. Therefore it is **important** to make sure the end-users can understand what your PR does, **based on your PR title**.
   - Other PRs (underlying development/intermediate implementation) should use tags with **everything in lowercase letters**: e.g. ``[metal], [opengl], [ir], [lang], [cli]``.
   - Because of the way the release changelog is generated, there should be **at most one captialized tag** in a PR title to prevent duplicate PR highlights. For example, ``[GUI] [Mac] Support modifier keys`` (#1189) is a bad example, we should use ``[gui] [Mac] Support modifier keys in GUI`` instead. Please capitalize the tag that is most relevant to the PR.


C++ and Python standards
------------------------

The C++ part of Taichi is written in C++17, and the Python part in 3.6+.
You can assume that C++17 and Python 3.6 features are always available.


Tips on the Taichi compiler development
---------------------------------------

:ref:`compilation` may worth checking out. It explains the whole compilation process.

See also :ref:`regress` if your work involves IR optimization.

When creating a Taichi program using ``ti.init(arch=desired_arch, **kwargs)``, pass in the following parameters to make the Taichi compiler print out IR:

- ``print_preprocessed = True``: print results of the frontend Python AST transform. The resulting scripts will generate a Taichi Frontend AST when executed.
- ``print_ir = True``: print the Taichi IR transformation process of kernel (excluding accessors) compilation.
- ``print_accessor_ir = True``: print the IR transformation process of data accessors, which are special and simple kernels. (This is rarely used, unless you are debugging the compilation of data accessors.)
- ``print_struct_llvm_ir = True``: save the emitted LLVM IR by Taichi struct compilers.
- ``print_kernel_llvm_ir = True``: save the emitted LLVM IR by Taichi kernel compilers.
- ``print_kernel_llvm_ir_optimized = True``: save the optimized LLVM IR of each kernel.
- ``print_kernel_nvptx = True``: save the emitted NVPTX of each kernel (CUDA only).

.. note::

  Data accessors in Python-scope are implemented as special Taichi kernels.
  For example, ``x[1, 2, 3] = 3`` will call the writing accessor kernel of ``x``,
  and ``print(y[42])`` will call the reading accessor kernel of ``y``.


Folder structure
----------------

Key folders are

- ``taichi``: The core compiler implementation

  - ``program``: Top-level constructs
  - ``ir``: Intermediate representation
  - ``analysis``: Static analysis passes
  - ``transforms``: IR transform passes
  - ``inc``: Small definition files to be included repeatedly

  - ``jit``: Just-In-Time compilation base classes
  - ``llvm``: LLVM utilities
  - ``runtime``: LLVM runtime environments

  - ``struct``: Struct compiler base classes
  - ``codegen``: Code generation base classes
  - ``backends``: Device-dependent code generators/runtime environments

    - ``cpu``: CPU backend implementation
    - ``cuda``: CUDA backend implementation
    - ``opengl``: OpenGL backend implementation
    - ``metal``: Metal backend implementation
    - ``cc``: C backend implementation (WIP)

  - ``gui``: GUI system
  - ``math``: Math utilities
  - ``python``: C++/Python interfaces

  - ``platform``: Platform supports
  - ``system``: OS-related infrastructure
  - ``util``:  Miscellaneous utilities


- ``python/taichi``: Python frontend implementation

  - ``core``: Loading & interacting with Taichi core
  - ``lang``: Python-embbed Taichi language & syntax (major)
  - ``misc``: Miscellaneous utilities
  - ``tools``: Handy end-user tools

- ``tests``: Functional tests

   - ``python``: Python tests (major)
   - ``cpp``: C++ tests

- ``examples``: Examples
- ``docs``: Documentation
- ``benchmarks``: Performance benchmarks
- ``external``: External libraries
- ``misc``: Random (yet useful) files
- ...




Testing
-------

Tests should be added to ``tests/``.

Command line tools
******************

- Use ``ti test`` to run all the tests.
- Use ``ti test -v`` for verbose outputs.
- Use ``ti test -C`` to run tests and record code coverage, see :ref:`coverage` for more infomations.

- Use ``ti test -a <arch(s)>`` for testing against specified backend(s).
  e.g. ``ti test -a cuda,metal``.

- Use ``ti test -na <arch(s)>`` for testing all architectures excluding some of them.
  e.g. ``ti test -na opengl,x64``.

- Use ``ti test <filename(s)>`` to run specific tests in filenames.
  e.g. ``ti test numpy_io`` will run all tests in ``tests/python/test_numpy_io.py``.

- Use ``ti test -c`` to run only the C++ tests.
  e.g. ``ti test -c alg_simp`` will run ``tests/cpp/test_alg_simp.cpp``.

- Use ``ti test -k <key>`` to run tests that match the specified key.
  e.g. ``ti test linalg -k "cross or diag"`` will run the ``test_cross`` and ``test_diag`` in ``tests/python/test_linalg.py``.

For more options, see ``ti test -h``.

For more details on how to write a test case, see :ref:`write_test`.

Documentation
-------------

Documentations are put under the folder ``docs/``.

- We use `reStructured text <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ (.rst) to write documentation.
- We host our documentation online using `readthedocs.io <https://taichi.readthedocs.io/en/stable>`_.
- Use ``ti doc`` to build the documentation locally.
- Open the documentation at ``docs/build/index.html``.

.. note::

  On Linux/OS X, use ``watch -n 1 ti doc`` to continuously build the documentation.

  If the OpenGL backend detector keeps creating new windows, execute ``export TI_WITH_OPENGL=0`` for ``ti doc``.


Efficient code navigation across Python/C++
-------------------------------------------
If you work on the language frontend (Python/C++ interface), to navigate around the code base, `ffi-navigator <https://github.com/tqchen/ffi-navigator>`_
allows you to jump from Python bindings to their definitions in C++.
Follow their README to set up your editor.


Upgrading CUDA
--------------

Right now we are targeting CUDA 10. When upgrading CUDA version,
the file ``external/cuda_libdevice/slim_libdevice.10.bc`` should also be replaced with a newer version.

To generate the slimmed version of libdevice based on a full ``libdevice.X.bc`` file from a CUDA installation,
use ``ti task make_slim_libdevice [libdevice.X.bc file]``
