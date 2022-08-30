---
sidebar_position: 1
slug: /
---


# Getting Started

Taichi is a high-performance parallel programming language using Python as the frontend. Taichi users write their computation-intensive tasks in Python obeying a few extra rules imposed by Taichi, and use the two decorators `ti.func` and `ti.kernel` to tell Taichi to take over the functions that implement the task. Taichi's just-in-time (JIT) compiler will compile these functions to machine code and all subsequent calls to them are executed on multi-CPU cores or GPUs. In typical compute-intense scenarios (such as numerical simulations), this will usually give a 50-100x speed up over native Python!

Taichi also has a built-in ahead-of-time (AOT) system that allows users to export the code as binary/shader files. These files can then be invoked in C/C++, without the Python environment.

## Requirements:

1. Python: 3.7/3.8/3.9/3.10 (64-bit)
2. OS: Windows (64-bit), OSX, Linux (64-bit)
3. GPUS: Cuda, Vulkan, OpenGL, Metal, dx11

## Installation

Taichi is available as a PyPI package:

```bash
pip install taichi
```
Taichi can also be built from the source, although we do not recommend this for first-time users except those who what to try the most up-to-date features. See our [developer's guide](../contribution/dev_install.md) for full details.

To verify the installation is successful, in the terminal run

```bash
ti gallery
```

This will pop up a window like follows:

<img style="margin:0px auto;display:block" width=480 src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/taichi-gallery.png"/>


Then click to choose and run the examples.

You can also run the command `ti example` to see the full list of examples included in the released package.

## Hello, world!

We introduce the Taichi programming language through a very basic fractal example, the [Julia fractal](https://en.wikipedia.org/wiki/Julia_set):

```python title=fractal.py
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

@ti.func
def complex_mul(z, w):  # complex multiplication of two 2d vectors
    return tm.vec2(z.x * w.x - z.y * w.y, z.x * w.y + z.y * w.x)

@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_mul(z, z) + c 
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

gui = ti.GUI("Julia Set", res=(n * 2, n))

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
```

You can run the above code by either saving it to your disk or directly running `ti example fractal` in the terminal, this will give you an animation:

<center>

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal.gif)

</center>

Let's dive into this simple Taichi program.


### import taichi

The first three lines

```python
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)
```

import taichi as a package as well as its `math` module and call the  `init` function to initialize some environment variables. The `math` module contains some frequently used math functions and built-in vector types of small dimensions. The `init` function accepts several arguments to allow users to custom the runtime program, for now, we only introduce the most important one, namely the `arch` . This argument will specify the *backend* that actually executes the compiled code. A backend can be either `ti.cpu` or  `ti.gpu`. For `ti.gpu` Taichi will look for GPU architectures in the order `ti.cuda`, `ti.vulkan`, `ti.opengl/ti.metal`, `ti.dx11` and choose the first available one. If no GPU device is found, Taichi will fall back to your CPU device. You can also directly specify the backend like `arch=ti.cuda`, Taichi will raise an error if this architecture is unavailable.


### Define a field

The next two lines

```python
n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))
```

defines a field of shape (640, 320) of float type. `field `  is the most important and frequently used data structure in Taichi. You can think of it as an analog of Numpy's `ndarray`  or Pytorch's `tensor`, but we emphasize here that Taichi's `field` is much more powerful and flexible than the other two counterparts: Taichi's field can be hierarchized in a tree-like manner, can store scalar, matrix, struct, and quant types, can be sparsely or dynamically allocated, can easily switch between the row-major/column-major modes, and has built-in automatic differentiation support. You will meet these features in more advanced tutorials later. For now, you can think of `pixels` just as a dense 2D floating array.


### Kernels and functions

Between lines 9-22 we defined two functions. One decorated by `ti.func` and one decorated by `ti.kernel`. Such functions are called *taichi functions* and *kernels* respectively, they are not executed by Python's virtual machine but will be taken over by Taichi's JIT compiler and get executed on the GPU.
    
The main differences between taichi functions and kernels are:
    
1. kernels are the entrances for Taichi to take over the subsequent task. Kernels can be called anywhere in your program, but taichi functions can only be called by kernels or by other taichi functions. In the above example, the taichi function `complex_mul` is called by the kernel `paint`.
2. The arguments and returns of a kernel function must all be type hinted, taichi functions do not have such restrictions. In the above example, the argument `t` in the kernel `paint` is type hinted, but the arguments `z,w`in the taichi function `complex_mul` are not.
3. Nested kernels are *not supported*, nested functions are *supported*. Recursively calling taichi functions are *not supported for now*.
    
:::tip
​    
For those who come from the world of CUDA, `ti.func` corresponds to `__device__` and `ti.kernel` corresponds to `__global__`.

For those who come from the world of OpenGL, `ti.func` corresponds to the usual function in GLSL and `ti.kernel` corresponds to a `compute shader`.
​:::


### Paralleled for loop

The real magic happens in line 15:

```python
for i, j in pixels:
```

This is a `for` loop at the outermost scope in a Taichi kernel and this loop is *automatically paralleled*. In fact, any `for` loop at the outermost scope in a kernel will be automatically paralleled. This is a very handy syntax sugar offered by Taichi. It allows users to parallel their tasks just in one plain and innocent loop, without bothering any underlying hooks like thread allocating/recycling and memory management!

Note that the field `pixels` is treated as an iterator, `i,j` are the indices of the elements and are integers in the range `[0,2*n-1]` and `[0,n-1]`, respectively. They are listed in the row-majored order `[0, 0]`, `[0, 1]`,...,  `[0, n-1]`,`[1,n-1]`, ... `[2*n-1, n-1]`.

Here we emphasize that *for loops not at the outermost scope will not be paralleled*, they are handled in serialized order:

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

`break` is *not* supported in parallel loops:

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

These should be enough for you to get prepared for exploring the more advanced features of Taichi.