---
sidebar_position: 1
slug: /
---


# Getting Started

Taichi is a high-performance parallel programming language embedded in Python.

Taichi users write their computation-intensive tasks in Python obeying a few extra rules imposed by Taichi, and use the two decorators `@ti.func` and `@ti.kernel` to ask Taichi to take over the functions that implement the task. Taichi's just-in-time (JIT) compiler will compile these functions to machine code and all subsequent calls to them are executed on multi-CPU cores or GPUs. In typical compute-intense scenarios (such as numerical simulations), this will usually lead to a 50-100x speed up over native Python!

Taichi's built-in ahead-of-time (AOT) system also allows you to export your code as binary/shader files, which can then be invoked in C/C++ and run without the Python environment. See [AOT deployment](../deployment/ndarray_android.md) for more details.

## Prerequisites

1. Python: 3.7/3.8/3.9/3.10 (64-bit)
2. OS: Windows, OS X, and Linux (64-bit)
3. Supported GPU backends (optional): CUDA, Vulkan, OpenGL, Metal, and DirectX 11

## Installation

Taichi is available as a PyPI package:

```bash
pip install taichi
```
You can also build Taichi from source: See our [developer's guide](../contribution/dev_install.md) for full details. We *do not* advise you to do so if you are a first-time user, unless you want to experience the most up-to-date features.

To verify a successful installation, run the following command in the terminal:

```bash
ti gallery
```

If Taichi is successfully installed, a window like the following image would pop up:

<img style="margin:0px auto;display:block" width=480 src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/taichi-gallery.png"/>


Then click to choose and run the examples.

You can also run the command `ti example` to view the full list of selected Taichi demos.

## Hello, world!

We would like to familiarize you with the Taichi programming language through a basic fractal example, the [Julia fractal](https://en.wikipedia.org/wiki/Julia_set):

```python title=fractal.py
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

@ti.func
def complex_sqr(z):  # complex square of a 2D vector
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(n * 2, n))

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
```

To run this program: Save the code above to your disk or execute the command `ti example fractal` directly in the terminal.
You will see the following animation:

<center>

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal.gif)

</center>

Let's dive into this simple Taichi program.


### Import Taichi

The first two lines

```python
import taichi as ti
import taichi.math as tm
```

import Taichi as a package as well as its `math` module. The `math` module contains some frequently used math functions and built-in vector and matrix types of small dimensions, such as `vec2` for 2D real vectors and `mat3` for 3x3 real matrices.

The line

```python
ti.init(arch=ti.gpu)
```

calls the `ti.init` function to initialize some environment variables. The `init` function accepts several arguments to allow users to custom the runtime program, for now, we only introduce the most important one, namely the `arch`.

The argument `arch` specifies the *backend* that actually executes the compiled code. A backend can be either `ti.cpu` or `ti.gpu`. For `ti.gpu` Taichi will look for GPU architectures in the order `ti.cuda`, `ti.vulkan`, `ti.opengl/ti.metal` and choose the first available one. If no GPU device is found, Taichi will fall back to your CPU device.

You can also directly specify the backend like `arch=ti.cuda`, Taichi will raise an error if this architecture is unavailable.


### Define a field

The next two lines

```python
n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))
```

define a field of shape (640, 320) of float type. `field` is the most important and frequently used data structure in Taichi. You can think of it as an analog of Numpy's `ndarray` or PyTorch's `tensor`, but we emphasize here that Taichi's `field` is a much more powerful and flexible data structure than the other two counterparts. For example, Taichi fields can be [spatially sparse](../basic/sparse.md), and can easily [switch between different data layouts](../basic/layout.md).

You will meet these features in more advanced tutorials later. You can now think of `pixels` as a dense 2D array.

### Kernels and functions

Between lines 9-22 we defined two functions. One decorated by `@ti.func` and one decorated by `@ti.kernel`. Such functions are called *Taichi functions* and *kernels* respectively. They are not executed by Python's interpreter but will be taken over by Taichi's JIT compiler and will execute rapidly on your parallel CPU cores or GPU.

The main differences between Taichi functions and kernels are:

1. Kernels are the entrances for Taichi to take over the subsequent task. Kernels can be called anywhere in your program, but Taichi functions can only be called by kernels or by other Taichi functions. In the above example, the Taichi function `complex_sqr` is called by the kernel `paint`.
2. The arguments and returns of a kernel function must all be type hinted, Taichi functions do not have such restrictions. In the above example, the argument `t` in the kernel `paint` is type hinted, but the argument `z` in the Taichi function `complex_sqr` is not.
3. Nested kernels are *not supported*, nested functions are *supported*. Recursively calling Taichi functions are *not supported for now*.

:::tip
​
For those who come from the world of CUDA, `ti.func` corresponds to `__device__` and `ti.kernel` corresponds to `__global__`.

For those who come from the world of OpenGL, `ti.func` corresponds to the usual function in GLSL and `ti.kernel` corresponds to a `compute shader`.
​:::


### Parallel for loop

The real magic happens at line 15:

```python
for i, j in pixels:
```

This is a `for` loop at the outermost scope in a Taichi kernel and this loop is *automatically paralleled*.

In fact, any `for` loop at the outermost scope in a kernel will be automatically paralleled. This is a very handy syntax sugar offered by Taichi. It allows users to parallel their tasks just in one plain and innocent loop, without bothering any underlying hooks like thread allocating/recycling and memory management!

Note that the field `pixels` is treated as an iterator, `i,j` are the indices of the elements and are integers in the range `[0, 2*n-1]` and `[0, n-1]`, respectively. They are listed in the row-majored order `(0, 0)`, `(0, 1)`, ..., `(0, n-1)`, `(1, n-1)`, ..., `(2*n-1, n-1)`.

Here we also emphasize that *for loops not at the outermost scope will not be paralleled*, they are handled in serialized order:

```python {3,7,14-15}
@ti.kernel
def fill():
    total = 0
    for i in range(10): # Paralleled
        for j in range(5): # Serialized in each parallel thread
            total += i * j

    if total > 10:
        for k in range(5):  # not paralleled since not at the outermost scope
```

:::caution WARNING

`break` statement is *not* supported in parallel loops:

```python
@ti.kernel
def foo():
    for i in x:
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

### Display the result

Lines 18-23 display the result in `pixels` to screen using Taichi's built-in [GUI system](../visualization/gui_system.md).

```python
gui = ti.GUI("Julia Set", res=(n * 2, n))

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
```

The line

```python
gui = ti.GUI("Julia Set", res=(n * 2, n))
```

sets the window title and the window resolution in pixels. Then in each round (we just render the animation 1000000 times) we compute the updated fractal pattern stored in `pixels`, call `gui.set_image` to set the window content as those in `pixels`, and call `gui.show()` to display the result to the screen.


### Summary

Congratulations! After walking through the above short example, you have learned the most significant features of Taichi:

1. It compiles and runs your kernel functions on backends.
2. Outermost `for` loops are automatically paralleled.
3. The field data container and how to loop over it.

These should prepare you well for more advanced features of Taichi!
