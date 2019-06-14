# The **Taichi** Programming Language
### High-Performance Differentiable Programming on Sparse Data Structures

# Installation
Supports Ubuntu 14.04/16.04/18.04, ArchLinux, Mac OS X. For GPU support, CUDA 9.0+ is needed.

 - Install `taichi` with the [installation script](https://taichi.readthedocs.io/en/latest/installation.html#ubuntu-arch-linux-and-mac-os-x)
 - Execute `ti install https://github.com/yuanming-hu/taichi_lang` to install the DSL project
 - Execute `python3 -m pip install astpretty astor opencv-python`
 - Execute `ti test` to run all the tests. It may take a few minutes.

# Basics
 - Every global variable is a tensor: `X[i, j, k]`. 0-d tensor (scalar) should be access as `X[None]`.
 - Differentiate element-wise product `*` and matrix product `@`.
 -	Debug your program with `ti.print(x)`.
 - Tensors values are initially zero.
 - Supported data types: `ti.i32`, `ti.i64`, `ti.f32`, `ti.f64`.
 - Binary operations on different types will give you a promoted type, e.g. i32 + f32 = f32.
 - Sparse tensors are initially inactive.
 
## Defining your kernels
 - Kernel arguments must be type hinted. Kernels can have at most 8 scalar parameters, e.g.
```python
@ti.kernel
def print(x: ti.i32, y: ti.f32):
  ti.print(x + y)
```
 - Restart the Taichi runtime system (clear memory, desctory all variables and kernels) : `ti.reset()`
 - Right now kernels can have either statements or at most one for loop.
 - `Taichi`-scope (`ti.kernel`) v.s. `Python`-scope: everything decorated by `ti.kernel` is in `Taichi`-scope, which will be compiled by the Taichi compiler.

# Differentiable Programming
 - No gradients are propagated to `int` tensors/locals
 - Remember to place your grad tensors, or use `ti.root.lazy_grad()`

# Performance tips
## Avoid synchronization
## Make Use of GPU Shared Memory and L1-d$
## Vectorization and Parallelization on CPUs

## Tweak your data structure
### Improve Cacheline Utilization
### Reduce Data Structure Overheads

# Sparsity
 - The user should make sure `grad` tensors have the same sparsity as the corresponding `primal` tensors.

# What’s different from TensorFlow
 - Imperative (instead of functional)
 - Fine-grained computation instead of large dense convolution on big tensors
 - Controllable (sparse) memory
 - Controllable gradient evaluation 
   - remember to clear the gradients!
## What do the grad kernels do
