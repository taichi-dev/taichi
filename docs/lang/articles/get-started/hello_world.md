---
sidebar_position: 1
---


# Hello, World!

Taichi is a domain-specific language embedded in Python and designed specifically for high-performance, parallel computing.

When writing compute-intensive tasks in Python, you can take advantage of Taichi's high performance computation by following a few extra rules. Generally, Taichi provides two decorators `@ti.func` and `@ti.kernel`, which instruct Taichi to take over the tasks. Its just-in-time (JIT) compiler compiles the decorated functions to machine code, and all subsequent calls to these functions are executed on multi-core CPUs or GPUs. In a typical compute-intensive scenario, such as a numerical simulation, Taichi can accelerate performance by 50x~100x compared with native Python code.

Taichi also has a built-in ahead-of-time (AOT) system for exporting your code into binary/shader files, which can then be called in C/C++ and run without the Python environment. See [Tutorial: Run Taichi programs in C++ application](../deployment/ndarray_android.md) for more information.

## Prerequisites

- Python: 3.7/3.8/3.9/3.10 (64-bit)
- OS: Windows, OS X, and Linux (64-bit)

## Installation

Taichi is available as a PyPI package:

```bash
pip install taichi
```

## Hello, world!

A basic fractal example, the [Julia fractal](https://en.wikipedia.org/wiki/Julia_set), can be a good starting point for you to understand the fundamentals of the Taichi programming language.

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

Save the code above to your local machine and run this program.


*You get the following animation:*

<center>

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/fractal.gif)

</center>

:::note

If you are *not* using an IDE for running code, you can simply run the code from your terminal by:

1. Go to the directory that contains your .py file
2. Type in `python3 *nameOfFile*.py` (replace *nameOfFile* with your programs name. Be sure to include the `.py` extension)

:::

Let's dive into this simple Taichi program.

### Import Taichi

The first two lines import Taichi and its `math` module. The `math` module contains built-in vectors and matrices of small dimensions, such as `vec2` for 2D real vectors and `mat3` for 3&times;3 real matrices.

See the [Math Module](../math/math_module.md) for more information.


### Initialize Taichi

:::note
`ti.init(**kwargs)`- Initializes Taichi environment and allows you to customize your Taichi runtime depending on the optional arguments passed into it.
:::

For now, we only introduce the most important argument, namely, `arch`.

The argument `arch` specifies the *backend* that executes the compiled code. A backend can be either `ti.cpu` or `ti.gpu`. When `ti.gpu` is specified, Taichi moves down the backend list of `ti.cuda`, `ti.vulkan`, and `ti.opengl/ti.metal`. If no GPU architecture is available, Taichi falls back to your CPU device.

You can also directly specify which GPU backend to use. For example, set `arch=ti.cuda` to run your program on CUDA. Taichi raises an error if the target architecture is unavailable. See the [Global Settings](../reference/global_settings.md) for more information about `ti.init()`.

```python{4}
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)
```

### Define a Taichi field

:::note
`ti.field(dtype, shape)` - Defines a Taichi field whose shape is of `shape` and whose elements are of type `dtype`.

`ti.field` is the most important and frequently used data structure in Taichi. You can compare it to NumPy's `ndarray` or PyTorch's `tensor`. But Taichi's field is more flexible. For example, a Taichi field can be [spatially sparse](../basic/sparse.md) and easily [switched between different data layouts](../basic/layout.md).

We will introduce more advanced features of fields in other scenario-based tutorials. For now, it suffices to know that the field `pixels` is a dense 2D array.

```python
n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))
```


### Kernels and functions

Lines 9~22 define two functions, one decorated with `@ti.func` and the other with `@ti.kernel`. They are called *Taichi function* and *kernel* respectively. Taichi functions and kernels are not executed by Python's interpreter but taken over by Taichi's JIT compiler and deployed to your parallel multi-core CPU or GPU.

The main differences between Taichi functions and kernels:

- Kernels are the entry points where Taichi kicks in and takes over the task. A kernel can be called anywhere, anytime in your program, while a Taichi function can be called only from inside a kernel or from inside another Taichi function. In the example above, the Taichi function `complex_sqr` is called by the kernel `paint`.
- A kernel *must* take type-hinted arguments and return type-hinted results. But Taichi functions do not require type hinting. In the example above, the argument `t` in the kernel `paint` is type hinted; the argument `z` in the Taichi function `complex_sqr` is not.
- Taichi *supports* nested functions but *does not support* nested kernels.
- Taichi *does not* support recursive Taichi function calls.

:::tip
​
For those who come from the world of CUDA, `ti.func` corresponds to `__device__` and `ti.kernel` corresponds to `__global__`.

For those who come from the world of OpenGL, `ti.func` corresponds to the usual function in GLSL and `ti.kernel` corresponds to a `compute shader`.

:::

### Parallel for loops

The key to high performance lies in how you iterate in Taichi. In particular, we can use parallelized looping to parse through our data more efficiently.

The following code snippet introduces a `for` loop at the outermost scope in a Taichi kernel and thus is *automatically parallelized*. Notice that the loop is also calling both `i` and `j` at the same time, and the program will run these iterations concurrently.

Taichi offers a handy syntax sugar: It parallelizes any `for` loop at the outermost scope in a kernel. This means that you can parallelize your tasks using one plain loop, without the need to know thread allocation/recycling or memory management.

Note that the field `pixels` is treated as an iterator. As the indices of the field elements, `i` and `j` are integers falling in the ranges `[0, 2*n-1]` and `[0, n-1]`, respectively. The pair `(i, j)` loops over the sets `(0, 0)`, `(0, 1)`, ..., `(0, n-1)`, `(1, n-1)`, ..., `(2*n-1, n-1)` in parallel.


```python
for i, j in pixels:
```


Keep in mind that the *for loops not at the outermost scope are not parallelized* but handled serially:

```python {3,7,14-15}
@ti.kernel
def fill():
    total = 0
    for i in range(10): # Parallelized
        for j in range(5): # Serialized in each parallel thread
            total += i * j

    if total > 10:
        for k in range(5):  # Not parallelized because it is not at the outermost scope
```

You can also serialize a for loop at the outermost scope using `ti.loop_config(serialize=True)`. See [Serialize a specified parallel for loop](../debug/debugging.md#serialize-a-specified-parallel-for-loop) for more information.

:::caution WARNING

The `break` statement is *not* supported in parallelized loops:

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

Lines 18~23 render the result on your screen using Taichi's built-in [GUI System](../visualization/gui_system.md):

```python
gui = ti.GUI("Julia Set", res=(n * 2, n))
# Sets the window title and the resolution

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()
```

The program iterates over `pixels` 1,000,000 times, and the fractal pattern stored in `pixels` is updated accordingly. Call `gui.set_image()` to set the window and `gui.show()` to display the synchronized result on your screen.

### Key takeaways

Congratulations! After walking through the short example above, you have learned the most significant features of Taichi:

- Taichi compiles and runs Taichi functions and kernels on the specified backend.
- For loops at the outermost scope in a Taichi kernel are automatically parallelized.
- Taichi provides a flexible data container field, and you can use indices to loop over a field.

## Taichi examples

The Julia fractal is one of the representative demos Taichi provides. To view more selected demos in Taichi gallery:

```bash
ti gallery
```

This window appears:

<center>

![image](https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi/taichi-gallery.png)

</center>

To access the full list of Taichi examples, run `ti example`. Other useful command lines:

- `ti example -p fractal`/`ti example -P fractal` prints the source code of the fractal example.
- `ti example -s fractal` saves the example to your current work directory.

## Supported systems and backends

The following table lists the supported operating systems and the backends that Taichi supports on these platforms:

| **platform** |      **CPU**       |      **CUDA**      |     **OpenGL**     |     **Metal**      |    **Vulkan**    |
| :----------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|   Windows    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |        N/A         | :heavy_check_mark: |
|    Linux     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |        N/A         | :heavy_check_mark: |
|    macOS     | :heavy_check_mark: |        N/A         |        N/A         | :heavy_check_mark: | :heavy_check_mark: |

- :heavy_check_mark:: supported;
- N/A: not available

You are now ready to move on and start writing your own Taichi programs. See the following documents for more information about how to use Taichi in some of its typical application scenarios:
- [Accelerate Python with Taichi](./accelerate_python.md)
- [Conduct Physical Simulation](./cloth_simulation.md)
- [Accelerate PyTorch with Taichi](./accelerate_pytorch.md).
