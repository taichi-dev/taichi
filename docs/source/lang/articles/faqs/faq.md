---
sidebar_position: 1
---

# Frequently Asked Questions

## Installation

### Why does my `pip` complain `package not found` when installing Taichi?

You may have a Python interpreter with an unsupported version. Currently, Taichi only supports Python 3.7/3.8/3.9/3.10 (64-bit) . For more information about installation-specific issues, please check [Installation Troubleshooting](./install.md).

## Parallel programming

### Outer-most loops in Taichi kernels are by default parallel. How can I **serialize** one of them?

A solution is to add an additional _ghost_ loop with only one iteration outside the loop you want to serialize.

```python {1}
for _ in range(1):  # This "ghost" loop will be "parallelized", but with only one thread. Therefore, the containing loop below is serialized.
    for i in range(100):  # The loop you want to serialize
        ...
```

### Does Taichi provide a barrier synchronization function similar to `__syncthreads()` or `glMemoryBarrier()`?

You can call `ti.sync()`, which is similar to CUDA's `cudaStreamSynchronize()`, in Taichi to synchronize the parallel for loops.

`__syncthreads()` is a block-level synchronization barrier, and Taichi provides a synonymous API `ti.simt.block.sync()`, which for now supports CUDA and Vulkan backends only. However, all block-level APIs are still experimental, and you should use this API only when it relates to SIMT operation synchronization and `SharedArray` reads and writes.

## Data structures

### How do I declare a field with a **dynamic length**?

The `dynamic` SNode supports variable-length fields. It acts similarly to `std::vector` in C++ or `list` in Python.

:::tip
An alternative solution is to allocate a large enough `dense` field, with a corresponding 0-D field
`field_len[None]` tracking its length. In practice, programs allocating memory using `dynamic`
SNodes may be less efficient than using `dense` SNodes, due to dynamic data structure
maintenance overheads.
:::

### How can I swap elements between two fields in the Taichi scope? `a,b = b,a` does not work.

Direct value assignments lead to semantic ambiguity. For example, `a = b` can mean data copy if `a` is pre-defined, or otherwise can serve to define and initialize `a`.

You can swap two fields in the Taichi scope using a struct for:

```python
a = ti.field(ti.i32, 32)
b = ti.field(ti.i32, 32)

@ti.func
def field_copy(src: ti.template(), dst: ti.template()):
    for I in ti.grouped(src):
        dst[I] = src[I]

@ti.kernel
def test():
    # copy b to a
    field_copy(b, a)
    print(a[0])

test()
```

### How do I compute the minimum/maximum of a field?

Use `ti.atomic_min/atomic_max` instead of `ti.min/max`. For example:

```python
x = ti.field(ti.f32, 32)

@ti.kernel
def x_min() -> ti.f32:
    ret: ti.f32 = x[0]

    for i in x:
        ti.atomic_min(ret, x[i])  # store result in the first argument

    return ret

x_min()
```

### Does Taichi support `bool` type?

Currently, Taichi does not support the `bool` type.

### How do I program on less structured data structures (such as graphs and tetrahedral meshes) in Taichi?

These structures have to be decomposed into 1D Taichi fields. For example, when representing a graph, you can allocate two fields, one for the vertices and the other for the edges. You can then traverse the elements using `for v in vertices` or `for v in range(n)`.

## Operations

### In Taichi v1.3.0, the matmul result of a vector and a transposed vector gives a scalar instead of a matrix.

Taichi distinguishes vectors from matrices starting from v1.3.0, as explained in the [release note](https://github.com/taichi-dev/taichi/releases/tag/v1.3.0). `transpose()` on a vector is no longer allowed. Use `a.outer_product(b)`, instead of `a @ b.transpose()`, to find the outer product of two vectors.

## Developement related

### Can I enable auto compeletion for Taichi?

Yes, Taichi's Python user-facing APIs should work natively with any language server for Python.

Take VSCode as an example, you can install `Python` or `Pylance` extensions to get language support like signature help with type information, code completion etc.

If it doesn't work out of box after installing the extension, please make sure the right Python interpreter is selected by:

- invoke command palette (`Shift + Command + P (Mac) / Ctrl + Shift + P (Windows/Linux)`)
- find `Python: Select Interpreter`
- make sure you select the path to the Python interpreter you're using with a `taichi` package installed

### How to install Taichi on a server without Internet access?

Follow these steps to install Taichi on a server without Internet access.

1. From a computer with Internet access, pip download Taichi, ensuring that this computer has the same operating system as the target server:

```plaintext
pip download taichi
```

_This command downloads the wheel package of Taichi and all its dependencies._

2. Copy the downloaded *.whl packages to your local server and install each with the following command. Note that you *must\* complete all dependency installation before installing Taichi.

```
python -m pip install xxxx.whl
```

## Integration with other libs/softwares

### What is the most convenient way to load images into Taichi fields?

One feasible solution is `field.from_numpy(ti.tools.imread('filename.png'))`.

### Can Taichi interact with **other Python packages** such as `matplotlib`?

Yes, Taichi supports many popular Python packages. Taichi provides helper functions such as `from_numpy` and `to_numpy` to transfer data between Taichi fields and NumPy arrays, so that you can also use your favorite Python packages (e.g., `numpy`, `pytorch`, `matplotlib`) together with Taichi as below:

```python
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

pixels = ti.field(ti.f32, (512, 512))

def render_pixels():
    arr = np.random.rand(512, 512)
    pixels.from_numpy(arr)   # load numpy data into taichi fields

render_pixels()
arr = pixels.to_numpy()  # store taichi data into numpy arrays
plt.imshow(arr)
plt.show()
import matplotlib.cm as cm
cmap = cm.get_cmap('magma')
gui = ti.GUI('Color map', (512, 512))

while gui.running:
    render_pixels()
    arr = pixels.to_numpy()
    gui.set_image(cmap(arr))
    gui.show()
```

Besides, you can also pass numpy arrays or torch tensors into a Taichi kernel as arguments. See [Interacting with external arrays](../basic/external.md) for more details.

### Can I integrate Taichi and Houdini?

The answer is an unequivocal Yes! Our contributors managed to embed [taichi_elements](https://github.com/taichi-dev/taichi_elements), a multi-material continuum physics engine, into Houdini as an extension, combining Houdini's flexibility in preprocessing with Taichi's strength in high-performance computation.

You can follow the instructions provided [here](https://github.com/taichi-dev/taichi_houdini).

## Precision related

### How do I accurately initialize a vector or matrix with `f64` precision when my default floating-point precision (`default_fp`) is `f32`?

To better understand the question, look at the program below:

```python
import taichi as ti

ti.init()

@ti.kernel
def foo():
    A = ti.Vector([0.2, 0.0], ti.f64)
    print('A =', A)

    B = ti.Vector([ti.f64(0.2), 0.0], ti.f64)
    print('B =', B)

foo()
```

You get the following output:

```
A = [0.200000002980, 0.000000000000]
B = [0.200000000000, 0.000000000000]
```

You may notice the value of `A` is slightly different from `[0.2, 0]`. This is because, by default, your float literals are converted to `ti.f32`, and `0.2` in `ti.f32` precision becomes `0.200000002980`. If you expect `A` and `B` to have `ti.f64` precision, use `ti.f64(0.2)` to preserve more effective digits here so that `0.2` keeps its `ti.f64` type.

Alternatively, if you can afford having all floating-point operations in `f64` precision, you can directly initialize Taichi with `ti.init(..., default_fp=ti.f64)`.

## From Python to Taichi

### Why does it always return an error when I pass a list from the Python scope to a Taichi kernel?

A Taichi kernel **cannot** take a Python list directly. You need to use NumPy arrays as a bridge.

For example, the following code snippet does not work:

```python skip-ci:Error
import taichi as ti
import numpy as np
ti.init()
x = ti.field(ti.i32, shape=3)
array = [10, 20, 30]

@ti.kernel
def test(arr: list):
    for i in range(3):
        x[i] = arr[i]
test(array)
```

You need to import NumPy:

```python
import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)
x = ti.field(ti.i32, shape=3)
array = np.array([10, 20, 30])
@ti.kernel
def test(arr: ti.types.ndarray()):
    for i in range(3):
        x[i] = arr[i]
test(array)
```

## Visualization

### Does the Taichi's GUI system support color mapping when rendering simulation results?

Taichi's GUI system can display colors when the field it accepts is a 3D vector field where each vector represents the RGB values of a pixel.

To enable color mapping, convert `ti.field` into a NumPy array and call Matplotlib's colormap (`cm`), as shown in the following example:

```python skip-ci:Trivial
pixels = ti.Vector.field(3, shape=(w, h))
gui = ti.GUI(f'Window title', (w, h))
step = 0
while gui.running: # Main loop
    simulate_one_substep(pixels)
    img = pixels.to_numpy()
    img = cm.jet(img)
    gui.set_image(img)
    gui.show()
```

## Objective-oriented programming

### Why does inheritance fail? I created a parent class and a child class, both decorated with `@ti.data_oriented`, and placed fields under `@ti.kernel`.

The problem does not lie with inheritance. All Taichi fields must be allocated/placed in the Python scope. In other words, you need to define a field before calling `@ti.kernel`.

For example, the following code snippet cannot run properly:

```python
@ti.data_oriented
class MyClass1():

    def __init__(self):
        self.testfield = ti.Vector.field(3, dtype=ti.f32)

    @ti.kernel
    def init_field(self):
        ti.root.dense(ti.i, 10).place(self.testfield)
```

Instead, refrain from involving `@ti.kernel` when declaring a field via `ti.root().place()`:

```python
@ti.data_oriented
class TriangleRasterizer:
    def __init__(self, n):
        self.n = n
        self.A = ti.Vector.field(2, dtype=ti.f32)
        self.B = ti.Vector.field(2, dtype=ti.f32)
        self.C = ti.Vector.field(2, dtype=ti.f32)
        self.c0 = ti.Vector.field(3, dtype=ti.f32)
        self.c1 = ti.Vector.field(3, dtype=ti.f32)
        self.c2 = ti.Vector.field(3, dtype=ti.f32)

        self.vertices = ti.root.dense(ti.i, n).place(self.A, self.B, self.C)
        self.colors = ti.root.dense(ti.i, n).place(self.c0, self.c1, self.c2)

        # Tile-based culling
        self.block_num_triangles = ti.field(dtype=ti.i32,
                                            shape=(width // tile_size,
                                                   height // tile_size))
        self.block_indicies = ti.field(dtype=ti.i32,
                                       shape=(width // tile_size,
                                              height // tile_size, n))
```

## From Taichi to Python

### How can I write data in Taichi fields to files? `write()` does not work.

You cannot save data in Taichi fields directly, but there is a workaround. Taichi allows interaction with external arrays. Use `to_numpy` to convert a Taichi field to a NumPy array, as explained in [this section](https://docs.taichi-lang.org/docs/master/external). Then write the Numpy array to files via `numpy.savetxt`.

A simple example:

```python
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

x = ti.field(dtype=ti.f32, shape= 10)
y = ti.Vector.field(n=2, dtype=ti.i32, shape=10)

@ti.kernel
def init():
    for i in x:
        x[i] = i * 0.5 + 1.0
    for i in y:
        y[i] = ti.Vector([i,i])

init()
np.savetxt('x.txt', x.to_numpy())
np.savetxt('y.txt', y.to_numpy())
```

And data in fields `x` and `y` can be found in files **x.txt** and **y.txt**, respectively.

### Why an image obtained using `field.to_numpy()` is rotated when displayed using `matplotlib`'s `plt.imshow()`?

Taichi fields adopt a different coordinate system from NumPy's arrays for storing images. In a Taichi field, [0,0] denotes the pixel at the lower left corner of the image; the first axis extends to the right of the image; the second axis extends to the top.

This is different from the usual convention taken by popular third-party libs like `matplotlib` or `opencv`, where [0, 0] denotes the pixel at the top left corner, the first axis extends down to the bottom of the image, and the second axis extends to the right.

Therefore, to display a NumPy array using `matplotlb`'s `imshow()`, you must rotate it 90 degrees clockwise.

## Miscellaneous

### How does Taichi compare with Python packages designed for data science or machine learning?

Popular packages designed for data science or machine learning include NumPy, JAX, PyTorch, and TensorFlow. A major difference between them and Taichi lies in the granularity of math operations.

A common feature shared by the other packages is that they treat a single data array as the smallest unit of operations. Take PyTorch as an example. PyTorch processes a tensor as a whole and thus prefers such operations as the addition/subtraction/multiplication/division of tensors and matrix multiplication. The operators are parallelized internally, but the implementation process is invisible to users. As a result, users have to combine operators in various ways if they want to manipulate elements in tensors.

Unlike them, Taichi makes element-level operations transparent and directly manipulates each iteration of the loops. This is why Taichi outperforms the other packages in scientific computing. In this sense, it compares more to C++ and CUDA.

### How does Taichi compare with Cython?

Cython is a superset of the Python language for quickly generating C/C++ extensions. It is a frequently-used tool to improve Python code performance thanks to its support for C data types and static typing. In fact, many modules in the official NumPy and SciPy code are written and compiled in Cython.

On the flip side, the mixture of Python and C values compromises Cython's readability. In addition, though Cython supports parallel computing to a certain degree (via multi-threading), it cannot offload computation to GPU backends.

Compared with Cython, Taichi is more friendly to Non-C users because it can achieve significant performance improvement with pure valid Python code. Supporting a wide range of backends, Taichi is subject to much fewer limits when performing parallel programming. In addition, unlike Cython, Taichi does not require the OpenMP API or an extra parallelism module to accelerate your program. Just specify a backend and wrap the loop with the decorator `@ti.kernel`; then, you can leave the job to Taichi.

### How does Taichi compare with Numba?

As its name indicates, Numba is tailored for NumPy. Numba is recommended if your functions involve vectorization of NumPy arrays. Compared with Numba, Taichi enjoys the following advantages:

- Taichi provides advanced features, including quantized data types, dataclasses and sparse data structures, and allows you to adjust memory layout flexibly. These features are especially helpful when a program handles massive amounts of data. However, Numba only performs best when dealing with dense NumPy arrays.
- Taichi can run on different GPU backends, making large-scale parallel programming (such as particle simulation or rendering) much more efficient. But it would be hard even to imagine writing a renderer in Numba.

### How does Taichi compare with ctypes?

ctypes allows you to call C/C++ compiled code from Python and run C++/CUDA programs in Python through a C-compatible API. It is a convenient option to access a vast collection of libraries in Python while achieving some improvement in performance. However, ctypes elevates the usage barrier: To write a satisfactory program, you need to command C, Python, CMake, CUDA, and even more languages. Moreover, ctypes may not fit in well with some performance-critical scenarios where you try to call large C libraries in Python, given the runtime overhead it incurs.

In contrast, it is much more reassuring to keep everything in Python. Taichi accelerates the performance of native Python code through automatic parallelization without involving the libraries out of the Python ecosystem. It also enables offline cache, which drastically reduces the launch overhead of Taichi kernels after the first call.

### How does Taichi compare with PyPy?

Similar to Taichi, PyPy also accelerates Python code via just-in-time (JIT) compilation. PyPy is attractive because users can keep Python scripts as they are without even moderate modification. On the other hand, its strict conformity with Python rules leaves limited room for optimization.

If you expect a greater leap in performance, Taichi can achieve the end. But you need to familiarize yourself with Taichi's syntax and assumptions, which differ from Python's slightly.
