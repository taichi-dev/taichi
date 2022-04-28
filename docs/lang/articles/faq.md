---
sidebar_position: 9999
---

# Frequently Asked Questions

### Why does my `pip` complain `package not found` when installing Taichi?

You may have a Python interpreter with an unsupported version. Currently, Taichi only supports Python 3.6/3.7/3.8 (64-bit) . For more information about installation related issues, please check [Installation Troubleshooting](./misc/install.md).

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

Besides, you can also pass numpy arrays or torch tensors into a Taichi kernel as arguments. See [Interacting with external arrays](./basic/external.md) for more details.

### How do I declare a field with a **dynamic length**?

The `dynamic` SNode supports variable-length fields. It acts similarly to `std::vector` in C++ or `list` in Python.

:::tip
An alternative solution is to allocate a large enough `dense` field, with a corresponding 0-D field
`field_len[None]` tracking its length. In practice, programs allocating memory using `dynamic`
SNodes may be less efficient than using `dense` SNodes, due to dynamic data structure
maintainance overheads.
:::

### How do I program on less structured data structures (such as graphs and tetrahedral meshes) in Taichi?

These structures have to be decomposed into 1D Taichi fields. For example, when representing a graph, you can allocate two fields, one for the vertices and the other for the edges. You can then traverse the elements using `for v in vertices` or `for v in range(n)`.
