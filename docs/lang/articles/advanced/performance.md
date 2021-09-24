---
sidebar_position: 5
---

# Performance tuning

## For-loop decorators

In Taichi kernels, for-loops in the outermost scope is automatically
parallelized.

However, there are some implementation details about **how it is
parallelized**.

Taichi provides some API to modify these parameters. This allows
advanced users to manually fine-tune the performance.

For example, specifying a suitable `ti.block_dim` could yield an almost
3x performance boost in
[examples/mpm3d.py](https://github.com/taichi-dev/taichi/blob/master/examples/mpm3d.py).

:::note
For performance profiling utilities, see [**Profiler** section of the Contribution Guide](../misc/profiler.md).
:::

### Thread hierarchy of GPUs

GPUs have a **thread hierarchy**.

From small to large, the computation units are: **iteration** \<
**thread** \< **block** \< **grid**.

- **iteration**: Iteration is the **body of a for-loop**. Each
  iteration corresponding to a specific `i` value in for-loop.
- **thread**: Iterations are grouped into threads. Threads are the
  minimal unit that is parallelized. All iterations within a thread
  are executed in **serial**. We usually use 1 iteration per thread
  for maximizing parallel performance.
- **block**: Threads are grouped into blocks. All threads within a
  block are executed in **parallel**. Threads within the same block
  can share their **block local storage**.
- **grid**: Blocks are grouped into grids. Grid is the minimal unit
  that being **launched** from host. All blocks within a grid are
  executed in **parallel**. In Taichi, each **parallelized for-loop**
  is a grid.

For more details, please see [the CUDA C programming
guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy).
The OpenGL and Metal backends follow a similar thread hierarchy.

### API reference

Programmers may **prepend** some decorator(s) to tweak the property of a
for-loop, e.g.:

```python
@ti.kernel
def func():
    for i in range(8192):  # no decorator, use default settings
        ...

    ti.block_dim(128)      # change the property of next for-loop:
    for i in range(8192):  # will be parallelized with block_dim=128
        ...

    for i in range(8192):  # no decorator, use default settings
        ...
```

# Local Storage Optimizations

Taichi comes with a few optimizations that leverages the *fast memory* (e.g. CUDA shared memory, L1 cache) for performance optimization.
The idea is straightforward: Where possible, Taichi substitues the accesses to the global memroy (slow) with that to the local one (fast), and writes the data in the local memory back to the global memory in the end. Such transformations preserves the semantics of the original program (will be explained later).

## Thread Local Storage (TLS)

TLS is mostly designed to optimize the parallel reduction. When Taichi identifies a global reduction pattern in a `@ti.kernel`, it automatically
applies the TLS optimization to make the generated code match those that are commonly found in the optimized GPU reduction implementations.

We will walkthrough an example using CUDA's terminology.

```python
x = ti.field(ti.f32, shape=1000000)
s = ti.field(ti.f32, shape=())

@ti.kernel
def sum():
  for i in x:
    s[None] += x[i]

sum()
```

Internally, Taichi's parallel loop is implemented using [Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).
What this means is that each physical CUDA thread could handle more than one item in `x`. That is, the number of threads launched for `sum` can be fewer
than the shape of `x`.

One optimization enabled by this strategy is to substitute the global memory access with a
*thread-local* one. Concretely, instead of directly and atomically adding `x[i]` into the
destination `s[None]`, which resides in the global memory, Taichi preallocates a thread-local
buffer upon entering the thread, accumulates (*non-atomically*) the value of `x` into this buffer, then adds the
result of the buffer back to `s[None]` atomically before exitting the thread. Assuming each
thread handles `N` items in `x`, the number of atomic adds is reduced to one-N-th its original size.

In addition, the last atomic add from the thread-local buffer to the global memory `s[None]` can be optimized
by using CUDA's warp-level intrinsics, further reducing the number of required atomic adds.

Currently, Taichi supports TLS optimization for these reduction operators: add, sub, min and max.

## Block Local Storage (BLS)

Context: For a sparse field with a hierarchical layout matching `ti.root.(sparse SNode)+.dense`, Taichi will assign
one CUDA thread block to each `dense` block. BLS optimization works specificially for such kinds of fields.

BLS aims to accelerate the stencil computation patterns by leveraging the CUDA shared memory. Roughly speaking, the BLS
optimization attempts to figure out the accessing range w.r.t the `dense` block at compile time. At runtime, the
generated code fetches all the data in range into a *block local* buffer (CUDA shared memory), and then reads the data
from this block local buffer instead of the global memory.

```python
a = ti.field(ti.f32)
# a's block size is 4x4
ti.root.pointer(ti.ij, 32).dense(ti.ij, 4).place(a)

@ti.kernel
def foo():
  # taichi will try to buffer `a` in CUDA shared memory
  ti.block_local(a)
  for i, j in a:
    print(a[i - 1, j], a[i, j + 2])
```

For a `dense` block in `a`, `i, j` has determined its range, i.e., `[N, N + 4) x [N, N + 4)`. 