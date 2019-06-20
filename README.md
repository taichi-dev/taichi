# The **Taichi** Programming Language
### High-Performance Differentiable Programming on Sparse Data Structures

# Installation
Supports Ubuntu 14.04/16.04/18.04, ArchLinux, Mac OS X. For GPU support, CUDA 9.0+ is needed.

 - Install `taichi` with the [installation script](https://taichi.readthedocs.io/en/latest/installation.html#ubuntu-arch-linux-and-mac-os-x)
 - Execute `ti install https://github.com/yuanming-hu/taichi_lang` to install the DSL project
 - Execute `python3 -m pip install astpretty astor opencv-python`
 - Add the following line to your `~/.bashrc` or `~/.zshrc` for the python frontend.
 ```bash
 export PYTHONPATH=$TAICHI_REPO_DIR/projects/taichi_lang/python:$PYTHONPATH
 ```
 - Execute `ti test` to run all the tests. It may take a few minutes.

# Global Tensors
 - Every global variable is an N-dimensional tensor. Global scalars are treated as 0-D tensors.
 - Global tensors are accessed using indices, e.g. `x[i, j, k]` if `x` is a 3D tensor. For 0-D tensor, access it as `x[None]`.
   - If you access a 0-D tensor `x` using `x = 0`, instead of `x[None] = 0`, the handle `x` will be set to zero instead of the value in that tensor. This is a compromise to the native python semantics. So please always use indexing to access entries in tensors.
 - Tensors values are initially zero.
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
```python
# Good kernels
@ti.kernel
def print(x: ti.i32, y: ti.f32):
  ti.print(x + y)
  ti.print(x * y)

@ti.kernel
def copy():
  for i in x:
    y[i] = x[i]
    
# Bad kernels that won't compile right now.
# (split them into two kernels for now. Compiler support coming soon.)
@ti.kernel
def print():
  ti.print(x + y)
  for i in x:
    y[i] = x[i]

@ti.kernel
def print():
  for i in x:
    y[i] = x[i]
  for i in x:
    z[i] = x[i]

```

- `Taichi`-scope (`ti.kernel`) v.s. `Python`-scope: everything decorated by `ti.kernel` is in `Taichi`-scope, which will be compiled by the Taichi compiler.

# Data layout
 - Non-power-of-two tensor dimensions are promoted into powers of two. For example, a tensor of size `(18, 65)` will be materialized as `(32, 128)`. Be careful if you want to iterate over this structural node when it is dense: the loop variables will become iterate over the promoted large domain instead of the original compact domain. Use a range-for instead. For sparse structural nodes, this makes no difference.

# Arithematics
 - Supported data types: `ti.i32`, `ti.i64`, `ti.f32`, `ti.f64`.
 - Binary operations on different types will give you a promoted type, e.g. `i32 + f32 = f32`.
 - `ti.Matrix` are for small matrices (e.g. `3x3`) only. If you have `64\times 64` matrices, you should consider using a 2D global tensor. `ti.Vector` is the same as `ti.Matrix`, except that it has only 1 column.
 - Differentiate element-wise product `*` and matrix product `@`.

# Differentiable Programming
 - No gradients are propagated to `int` tensors/locals
 - Remember to place your grad tensors, or use `ti.root.lazy_grad()`
 - The user should make sure `grad` tensors have the same sparsity as the corresponding `primal` tensors.
 - Reset gradients every time.

# Debugging
 -    Debug your program with `ti.print(x)`.

# Performance tips
## Avoid synchronization
 - When using GPU, an asynchronous task queue will be maintained. Whenever reading/writing global tensors, a synchronization will be invoked, which leads to idle cycles on CPU/GPU. 
## Make Use of GPU Shared Memory and L1-d$
 - 
## Vectorization and Parallelization on CPUs
 - 

## Tweaking your data structure
### Improve Cacheline Utilization
### Reduce Data Structure Overheads

# Sparsity

# What’s different from TensorFlow
 - Imperative (instead of functional)
 - Fine-grained computation instead of large dense convolution on big tensors
 - Controllable (sparse) memory
 - Controllable gradient evaluation 
   - remember to clear the gradients!
## What do the grad kernels do
 - 
