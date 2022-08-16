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

1. From a computer with Internet access, download the installation packages for dill, sourceinspect, and taichi, ensuring that this computer has the same operating system as the target server:

```plaintext
pip download dill sourceinspect taichi
```

2. Copy all the downloaded `*.whl` packages to your local server.

3. On your local server, change directory to **/home/<user_name>/<taichi_tmp>/** and install Taichi using the following command:

```plaintext
pip install ./*
```

### Can I integrate Taichi and Houdini?

The answer is an unequivocal Yes! Our contributor managed to embed [Taichi-element](https://github.com/taichi-dev/taichi_elements), a multi-material continuum physics engine, into Houdini as an extension, combining Houdini's flexibility in preprocessing with Taichi's strength in high-performance computation.

You can follow the instructions provided [here](https://github.com/taichi-dev/taichi_houdini).

### How do I prevent precision loss in situations where 32-bit precision is required?

If you use the default `ti.f32`, then this is normal because `float32` has only six to seven decimal digits of precision. Even a constant is cast to a `float32` number by default. To solve this issue, declare `default_fp=ti.f64` in `ti.init`.

For example, if you stay with `float32`:

```python
import taichi as ti
ti.init(debug=True)
@ti.func
def get_f_stress3(f_stress):
    res = ti.Matrix([[f_stress[0], f_stress[2], 0.0], [f_stress[2], f_stress[1], 0.0], [0.0, 0.0, f_stress[3]]], ti.f64)
    return res
@ti.kernel
def foo():
    stress = get_f_stress3(f_stress)
    print("σ =", stress)
if __name__ == "__main__":
    f_stress = ti.Vector([-0.742128, -1.731632, -0.000001, -0.742128], ti.f64)
    foo()
```

The output becomes:

 ```plaintext
 [Taichi] version 1.0.0, llvm 10.0.0, commit 6a15da85, win, python 3.8.1
[Taichi] Starting on arch=x64
σ = [[-0.742128014565, -0.000001000000, 0.000000000000], [-0.000001000000, -1.731631994247, 0.000000000000], [0.000000000000, 0.000000000000, -0.742128014565]]
 ```

After switching to `float64` by setting `ti.init(debug=True, default_fp=ti.f64)`, you get the correct output:

```plaintext
σ = [[-0.742128000000, -0.000001000000, 0.000000000000], [-0.000001000000, -1.731632000000, 0.000000000000], [0.000000000000, 0.000000000000, -0.742128000000]]
 ```

### Could you recommend some reading materials for cloth simulation, or about multigrid methods in particular?

[This paper](https://www.cs.cmu.edu/~baraff/papers/sig98.pdf) proposes an implicit integration method that enforces constraints on individual cloth particles.

[This paper](https://www.cs.ubc.ca/~rbridson/docs/cloth2002.pdf) presents a collision handling algorithm compatible with any technique that simulates the internal dynamics of a piece of cloth.

As for multigrid methods, you may be interested in an efficient multigrid scheme, which simulates high-resolution deformable objects in their full spaces at interactive frame rates, as explained in [this paper](http://tiantianliu.cn/papers/xian2019multigrid/xian2019multigrid.html).

### Why does it always return an error when I pass a list from the Python scope to a Taichi kernel?

A Taichi kernel cannot take a Python list directly. You need to use NumPy Arrays as a bridge.

For example, the following code snippet does not work:

```python
import taichi as ti
ti.init()
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a: return i
    return -1
@ti.kernel
def sort_by_values(list1: list, values: list):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list
l1 = [0, 1, 2, 3, 4, 5]
vs = [225, 114, 123, 53, 654, 15]
r = sort_by_values(l1, vs)
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
