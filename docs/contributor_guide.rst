Contribution guidelines (WIP)
===============================================

First of all, thank you for contributing! We welcome contributions of
all forms, including but not limited to

- Bug fixes
- New features
- Documentation
- New example programs
- Compiler performance patches
- Minor typo fixes in the documentation, code, comments (please directly make a pull request for minor issues like these)

How to contribute bug fixes and new features
--------------------------------------------------

If you are interested in resolving an issue marked with `"welcome contribution" <https://github.com/taichi-dev/taichi/issues?q=is%3Aopen+is%3Aissue+label%3A%22welcome+contribution%22>`_.
Please first leave a note (e.g. *I know how to fix this and would like to help!*) on the issue, so that
people know some one is already working on it.

If no lead developer has commented and described a potential solution on the issue, please also briefly
describe your plan and wait for a lead developer to reply before you start.

Following this rule will prevent contributors from doing redundant work,
while keeping the solution simple and effective.

Tips on Taichi compiler development
--------------------------------------------------

:ref:`compilation` may worth checking out. It explains the whole compilation process.

Set ``ti.get_runtime().print_preprocessed = True``
to inspect results of the frontend Python AST transform. The resulting scripts will generate a Taichi Frontend AST when executed.

Set ``ti.cfg.print_ir = True`` to inspect the IR transformation process of compilation.

Set ``ti.cfg.print_kernel_llvm_ir = True`` to inspect the emitted LLVM IR for each invoked kernel.

