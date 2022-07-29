---
sidebar_position: 1
slug: /
---

# Getting Started

## Target audience

End users who *only* wish to quickly set up Taichi for simulation or high-performance numerical computation.

:::caution IMPORTANT

For developers who are interested in the compiler, computer graphics, or high-performance computing, and would like to contribute new features or bug fixes to the [Taichi programming language](https://github.com/taichi-dev/taichi), see the [Developer installation](../contribution/dev_install.md) for more information on building Taichi from source.

:::

## Prerequisites

### Python

3.6/3.7/3.8/3.9/3.10 (64-bit)

:::note

Taichi recommends installing Python from [Miniforge](https://github.com/conda-forge/miniforge/#download) conda if you are on a MacBook with M1 chip.

:::

### Supported systems and backends

The following table lists the supported operating systems and the backends that Taichi supports on these platforms:

| **platform** |      **CPU**       |      **CUDA**      |     **OpenGL**     |     **Metal**      |    **Vulkan**    |
| :----------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|   Windows    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |        N/A         | :heavy_check_mark: |
|    Linux     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |        N/A         | :heavy_check_mark: |
|    macOS     | :heavy_check_mark: |        N/A         |        N/A         | :heavy_check_mark: | :heavy_check_mark: |

- :heavy_check_mark:: supported;
- N/A: not available

## Installation

To get started with the Taichi Language, simply install it with `pip`:

```shell
python3 -m pip install taichi
```

There are a few of extra requirements depend on which operating system you are using:

````mdx-code-block

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="arch-linux"
  values={[
    {label: 'Arch Linux', value: 'arch-linux'},
    {label: 'Windows', value: 'windows'},
  ]}>

  <TabItem value="arch-linux">

  On Arch Linux, you need to install `ncurses5-compat-libs` package from the Arch User Repository: `yaourt -S ncurses5-compat-libs`

  </TabItem>
  <TabItem value="windows">

  On Windows, please install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) if you haven't done so.

  </TabItem>
</Tabs>
````

See the [Installation Troubleshooting](../faqs/install.md) if you run into any issues when installing Taichi.

A successful installation of Taichi should add a CLI (Command-Line Interface) to your system, which is helpful to perform several routine tasks quickly. To invoke the CLI, please run `ti` or `python3 -m taichi`.

## Examples

Taichi provides a set of bundled examples. You could run `ti example -h` to print the help message and get a list of available example names.

For instance, to run the basic `fractal` example, try: `ti example fractal` from your shell. (`ti example fractal.py` should also work)

You may print the source code of example by running `ti example -p fractal`, or `ti example -P fractal` for print with syntax highlight.

You may also save the example to current work directory by running `ti example -s fractal`.

## Hello, world!

We introduce the Taichi programming language through a very basic _fractal_ example.

Running the Taichi code below using either `python3 fractal.py` or `ti example fractal` will give you an animation of [Julia set](https://en.wikipedia.org/wiki/Julia_set):

<center>


![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal.gif)

</center>

```python title=fractal.py
import taichi as ti

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])

@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
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

Let's dive into this simple Taichi program.

### import taichi as ti

Taichi is a domain-specific language (DSL) embedded in Python.

To make Taichi as easy-to-use as a Python package, we have done heavy engineering with this goal in mind - letting every Python programmer write Taichi programs with minimal learning effort.

You can even use your favorite Python package management system, Python IDEs and other Python packages together with Taichi.

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
