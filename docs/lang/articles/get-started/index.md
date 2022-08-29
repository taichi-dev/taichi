---
sidebar_position: 1
slug: /
---

# Getting Started
---
sidebar_position: 1
slug: /
---

# Getting Started

Taichi is a high-performance parallel programming language using Python as the frontend. The commom usage is, users write their computation-intensive tasks in python obeying a few extra rules imposed by Taichi and use two decorators `ti.func` and `ti.kernel` to tell Taichi to take over the functions that implement the task. Taichi's just-in-time compiler will compile these functions to machine code and all subsequent calls to them are executed on multi-CPU cores or GPU. In typical scenarios like physical simulations or real-time renderings, this will usually give a 50~100x speed up compared to native python!

Taichi also has a built-in ahead-of-time compiling module that allows users to export the code as shader files that can be called out of the python environment.


## Prerequisites

1. Python: 3.7/3.8/3.9/3.10 (64-bit)
2. OS: Windows (64-bit), OSX, Linux (64-bit)
3. GPUS: Cuda, Vulkan, OpenGL, Metal


## Installation

Taichi is available as a PyPI package:

```bash
pip install taichi
```
Taichi can also be built from the source, although we do not recommend this for first-time users except those who what to try the most up-to-date features. See our [eveloper's guide](../contribution/dev_install.md) for full details.

To verify the installation is successful, in terminal run

```bash
ti gallery
```

This will pop up a window like follows:

<center>

![image](https://github.com/taichi-dev/taichi_assets/blob/master/static/imgs/ti_gallery.png)

</center>

Click to choose and run the examples.

You can also run the command `ti example` to see the full list of examples included in the released package.

## Hello, world!

We introduce the Taichi programming language through a very basic fractal example, the [Julia fractal](https://en.wikipedia.org/wiki/Julia_set):

```python title=fractal.py
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = tm.cmul(z, z) + c  # cmul is the complex multiplicaiton of two vec2s
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(n * 2, n))

i = 0
while gui.running:
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
    i = i + 1
```

You can run the above code by either save it to your disk or directly run `ti example fractal` in a terminal, this will give you an animation:

<center>

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal.gif)

</center>

Let's dive into this simple Taichi program.

### import taichi as ti

```python
# Initialize Taichi and run it on CPU (default)
# - `arch=ti.gpu`: Run Taichi on GPU and has Taichi automatically detect the suitable backend
# - `arch=ti.cuda`: For the NVIDIA CUDA backend
# - `arch=ti.metal`: [macOS] For the Apple Metal backend
# - `arch=ti.opengl`: For the OpenGL backend
# - `arch=ti.vulkan`: For the Vulkan backend
# - `arch=ti.dx11`: For the DX11 backend
ti.init(arch=ti.cpu)
```

:::info

- With `arch=ti.gpu`, Taichi first tries to run on CUDA. If CUDA is not supported on your machine, Taichi falls back on Metal, OpenGL, Vulkan, or DX11.
- If no GPU backend (CUDA, Metal, OpenGL, Vulkan, or DX11) is supported, Taichi falls back to the CPU backend.
:::

:::note

When running on the CUDA backend on Windows or ARM devices (for example NVIDIA Jetson), Taichi allocates 1 GB GPU memory for field storage by default.

To override this behavior, do either of the following:

- Initialize Taichi using `ti.init(arch=ti.cuda, device_memory_GB=3.4)` to allocate `3.4` GB GPU memory,
- Initialize Taichi using `ti.init(arch=ti.cuda, device_memory_fraction=0.3)` to allocate `30%` of the GPU memory.

On platforms other than Windows or ARM devices, Taichi relies on its on-demand memory allocator to allocate memory adaptively.
:::

### Fields

Taichi is a *data*-oriented programming language where dense or spatially-sparse fields are the first-class citizens.

In the code above, `pixels = ti.field(dtype=float, shape=(n * 2, n))` allocates a 2D dense field named `pixels` of size `(640, 320)` and element data type `float`.

### Functions and kernels

Computation resides in Taichi **kernels** and Taichi **functions**.

Taichi **kernels** are defined with the decorator `@ti.kernel`. They can be called from Python to perform computation. Kernel arguments must be type-hinted (if any).

Taichi **functions** are defined with the decorator `@ti.func`. They can *only* be called by Taichi kernels or other Taichi functions.

See [syntax](../kernels/syntax.md) for more details about Taichi kernels and functions.

The language used in Taichi kernels and functions looks exactly like Python, yet the Taichi frontend compiler converts it into a language
that is **compiled, statically-typed, lexically-scoped, parallel and differentiable**.

:::info

*Taichi scope vs. Python scope*:

Everything decorated with `@ti.kernel` and `@ti.func` is in the Taichi scope and hence will be compiled by the Taichi compiler.
Everything else is in the Python scope. They are simply Python native code.
:::

:::caution

- Taichi kernels must be called from the Python-scope.
- Taichi functions must be called from the Taichi-scope.
:::

:::tip

For those who come from the world of CUDA, `ti.func` corresponds to `__device__` while `ti.kernel` corresponds to `__global__`.
:::

:::note
- Nested kernels are *not supported*.
- Nested functions are *supported*.
- Recursive functions are *not supported for now*.
:::

### Parallel for-loops

For loops at the outermost scope in a Taichi kernel is *automatically parallelized*. For loops can have two forms:

- _range-for loops_
- _struct-for loops_.

*Range-for loops* are no different from the Python for loops, except that they are parallelized when used at the outermost scope. Range-for loops can be nested.

```python {3,7,14-15}
@ti.kernel
def fill():
    for i in range(10): # Parallelized
        x[i] += i
        s = 0
        for j in range(5): # Serialized in each parallel thread
            s += j
        y[i] = s

@ti.kernel
def fill_3d():
    # Parallelized for all 3 <= i < 8, 1 <= j < 6, 0 <= k < 9
    for i, j, k in ti.ndrange((3, 8), (1, 6), 9):
        x[i, j, k] = i + j + k
```

:::note

It is the loop *at the outermost scope* that gets parallelized, not the outermost loop.

```python
@ti.kernel
def foo():
    for i in range(10): # Parallelized :-)
        ...

@ti.kernel
def bar(k: ti.i32):
    if k > 42:
        for i in range(10): # Serial :-(
            ...
```
:::

*Struct-for loops* are particularly useful when iterating over (sparse) field elements. In the `fractal.py` above, `for i, j in pixels` loops over all the pixel coordinates, i.e.,`(0, 0), (0, 1), (0, 2), ... , (0, 319), (1, 0), ..., (639, 319)`.

:::note
Struct-for is the key to [sparse computation](../basic/sparse.md) in Taichi, as it will only loop over active elements in a sparse field. In dense fields, all elements are active.
:::

:::caution WARNING

Struct-for loops must live at the outer-most scope of kernels.
It is the loop **at the outermost scope** that gets parallelized, not the outermost loop.

```python
x = [1, 2, 3]

@ti.kernel
def foo():
    for i in x: # Parallelized :-)
        ...

@ti.kernel
def bar(k: ti.i32):
    # The outermost scope is a `if` statement
    if k > 42:
        for i in x: # Not allowed. Struct-fors must live in the outermost scope.
            ...
```
:::

:::caution WARNING

`break` is *not* supported in parallel loops:

```python
@ti.kernel
def foo():
    for i in x:
        ...
        break # Error!

    for i in range(10):
        ...
        break # Error!

@ti.kernel
def foo():
    for i in x:
        for j in range(10):
            ...
            break # OK!
```
:::

### GUI system

Taichi provides a CPU-based [GUI system](../visualization/gui_system.md) for you to render your results on the screen.

```python
gui = ti.GUI("Julia Set", res=(n * 2, n))

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
```

## Still have issues?

If you encounter any issue that is not covered here, feel free to report it by [opening an issue on GitHub](https://github.com/taichi-dev/taichi/issues/new?labels=potential+bug&template=bug_report.md) and including the details. We are always there to help!

See the [Installation Troubleshooting](../faqs/install.md) if you run into any issues when installing Taichi.
