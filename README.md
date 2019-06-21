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

# What’s different from existing frameworks
 - **Sparsity**. The first-class sparse data structures makes it possible to develop efficient. 
 - **Imperative**.
 - **Fine-grainularity**.
   A typical data operation granulatity is `128x27x27x128`. instead of large dense convolution on big tensors
 - Controllable (sparse) memory
 - Controllable gradient evaluation 
   - remember to clear the gradients!
 - **High customizability**
 - **Auto diff**
## Comparison with TensorFlow and PyTorch
 - 


## What do the grad kernels do

# Multi-Stage Programming
 - Use `ti.static` for compile-time branching (For those who come from C++17, this is [`if constexpr`](https://en.cppreference.com/w/cpp/language/if).)
```python
enable_projection = True

@ti.kernel
def static():
  if ti.static(enable_projection): # No runtime overhead
    x[0] = 1
```
 - Use `ti.static` for forced loop unrolling
```python
@ti.kernel
def g2p(f: ti.i32):
  for p in range(0, n_particles):
    base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
    fx = x[f, p] * inv_dx - ti.cast(base, real)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0),
         0.5 * ti.sqr(fx - 0.5)]
    new_v = ti.Vector([0.0, 0.0])
    new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

    # Unrolled 9 iterations for higher performance
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        dpos = ti.cast(ti.Vector([i, j]), real) - fx
        g_v = grid_v_out[base(0) + i, base(1) + j]
        weight = w[i](0) * w[j](1)
        new_v += weight * g_v
        new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx

    v[f + 1, p] = new_v
    x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
    C[f + 1, p] = new_C
```
# Python Frontend
Embedding the language in `python` has the following advantages:
- Easy to learn. Python itself is very easy to learn, so is PyTaichiLang.
- Easy to run. No ahead-of-time compilation is needed.
- It allows people to reuse existing python infrastructure: 
   - IDEs. A python IDE simply works for TaichiLang, with syntax highlighting, checking, and autocomplete.
   - Package manager (pip). A developed Taichi application and be easily submitted to `PyPI` and others can easily set it up   with `pip`.
   - Existing packages. Interacting with other python components is just trivial.
- The built-in AST manipulation tools in `python` allow us to do magical things, as long as the kernel body can be parsed by the `python` parser.

However, this design decision has drawbacks as well:
 - Indexing is always needed when accessing elements in tensors, even if the tensor is 0D. Use `x[None] = 123` to set the value in `x` if `x` is 0D. This is because `x = 123` will set `x` itself (instead of its containing value) to be the constant `123` in python syntax, and unfortunately we cannot modify this behavior. 
 - When dealing with local matrices, syntax like `x(0, 1).val = y` is needed. It would be ideal to write `x(0, 1) = y`, but in `python` assigning to function call is not allowed. For global matrices you can use `x(0, 1)[i, j, k] = 42` and no special attention is needed.
