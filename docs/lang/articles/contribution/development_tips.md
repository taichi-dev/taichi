---
sidebar_position: 5
---

# Development Tips

This section gives you some tips on the Taichi compiler development.
Please make sure you have gone through [developer installation](./dev_install.md).

## Workflow of the Taichi compiler

[Life of a Taichi kernel](../internals/compilation.md) is a good place to get started,
which explains the whole compilation process step by step.

## C++ and Python standards

The C++ part of the Taichi compiler is written in C++17, and the Python part in 3.7+.
You can assume that C++17 and Python 3.7 features are always available.

## Efficient code navigation across Python/C++

If you are working on the language frontend (Python/C++ interface), you may
want to navigate across Python/C++ code. [ffi-navigator](https://github.com/tqchen/ffi-navigator)
allows you to jump from Python bindings to their definitions in C++. Please follow their
README to set up your editor.

## Printing IRs in different stages

When creating a Taichi program using
`ti.init(arch=desired_arch, **kwargs)`, pass in the following parameters
to make the Taichi compiler print out IRs in different stages:

- `print_ir=True`: print the Taichi IR transformation process of
  kernel (excluding accessors) compilation.
- `print_accessor_ir=True`: print the IR transformation process of
  data accessors, which are special and simple kernels. This is
  rarely used, unless you are debugging the compilation of data
  accessors.
- `print_struct_llvm_ir=True`: save the emitted LLVM IR by Taichi
  struct compilers.
- `print_kernel_llvm_ir=True`: save the emitted LLVM IR by Taichi
  kernel compilers.
- `print_kernel_llvm_ir_optimized=True`: save the optimized LLVM IR
  of each kernel.
- `print_kernel_nvptx=True`: save the emitted NVPTX of each kernel
  (CUDA only).

:::note
Data accessors in Python-scope are implemented as special Taichi
kernels. For example, `x[1, 2, 3] = 3` will call the writing accessor
kernel of `x`, and `print(y[42])` will call the reading accessor kernel
of `y`.
:::
