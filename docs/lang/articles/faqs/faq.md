---
sidebar_position: 1
---

# Frequently Asked Questions

### Why does my `pip` complain `package not found` when installing Taichi?

You may have a Python interpreter with an unsupported version. Currently, Taichi only supports Python 3.7/3.8/3.9/3.10 (64-bit) . For more information about installation-specific issues, please check [Installation Troubleshooting](./install.md).

### Does Taichi provide built-in constants such as `ti.pi`?

There is no built-in constant such as `pi`. We recommended using `math.pi` directly.

### Outer-most loops in Taichi kernels are by default parallel. How can I **serialize** one of them?

A solution is to add an additional *ghost* loop with only one iteration outside the loop you want to serialize.

```python {1}
for _ in range(1):  # This "ghost" loop will be "parallelized", but with only one thread. Therefore, the containing loop below is serialized.
    for i in range(100):  # The loop you want to serialize
        ...
```

### What is the most convenient way to load images into Taichi fields?

One feasible solution is `field.from_numpy(ti.tools.imread('filename.png'))`.

### Can Taichi interact with **other Python packages** such as `matplotlib`?

Yes, Taichi supports many popular Python packages. Taichi provides helper functions such as `from_numpy` and `to_numpy` to transfer data between Taichi fields and NumPy arrays, so that you can also use your favorite Python packages (e.g., `numpy`, `pytorch`, `matplotlib`) together with Taichi as below:

```python
import taichi as ti
pixels = ti.field(ti.f32, (1024, 512))
import numpy as np
arr = np.random.rand(1024, 512)
pixels.from_numpy(arr)   # load numpy data into taichi fields
import matplotlib.pyplot as plt
arr = pixels.to_numpy()  # store taichi data into numpy arrays
plt.imshow(arr)
plt.show()
import matplotlib.cm as cm
cmap = cm.get_cmap('magma')
gui = ti.GUI('Color map')
while gui.running:
    render_pixels()
    arr = pixels.to_numpy()
    gui.set_image(cmap(arr))
    gui.show()
```

Besides, you can also pass numpy arrays or torch tensors into a Taichi kernel as arguments. See [Interacting with external arrays](../basic/external.md) for more details.

### How do I declare a field with a **dynamic length**?

The `dynamic` SNode supports variable-length fields. It acts similarly to `std::vector` in C++ or `list` in Python.

:::tip
An alternative solution is to allocate a large enough `dense` field, with a corresponding 0-D field
`field_len[None]` tracking its length. In practice, programs allocating memory using `dynamic`
SNodes may be less efficient than using `dense` SNodes, due to dynamic data structure
maintenance overheads.
:::

### How do I program on less structured data structures (such as graphs and tetrahedral meshes) in Taichi?

These structures have to be decomposed into 1D Taichi fields. For example, when representing a graph, you can allocate two fields, one for the vertices and the other for the edges. You can then traverse the elements using `for v in vertices` or `for v in range(n)`.

### How to install Taichi on a server without Internet access?

Follow these steps to install Taichi on a server without Internet access.

1. From a computer with Internet access, pip download Taichi, ensuring that this computer has the same operating system as the target server:

```plaintext
pip download taichi
```

*This command downloads the wheel package of Taichi and all its dependencies.*

2. Copy the downloaded *.whl packages to your local server and install each with the following command. Note that you *must* complete all dependency installation before installing Taichi.

```
python -m pip install xxxx.whl
```

### Can I integrate Taichi and Houdini?

The answer is an unequivocal Yes! Our contributors managed to embed [taichi_elements](https://github.com/taichi-dev/taichi_elements), a multi-material continuum physics engine, into Houdini as an extension, combining Houdini's flexibility in preprocessing with Taichi's strength in high-performance computation.

You can follow the instructions provided [here](https://github.com/taichi-dev/taichi_houdini).

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

### Why does it always return an error when I pass a list from the Python scope to a Taichi kernel?

A Taichi kernel **cannot** take a Python list directly. You need to use NumPy arrays as a bridge.

For example, the following code snippet does not work:

```python
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
