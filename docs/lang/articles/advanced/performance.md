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
The idea is straightforward: Where possible, Taichi substitues the accesses to the global memroy (slow) with that to the local one (fast),
and writes the data in the local memory back to the global memory in the end. Such transformations preserves the semantics of the original program (will be explained later).

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

Additionally, the last atomic add to the global memory `s[None]` is optimized using CUDA's warp-level intrinsics,
further reducing the number of required atomic adds.

Currently, Taichi supports TLS optimization for these reduction operators: add, sub, min and max. [Here](https://github.com/taichi-dev/taichi/pull/2956)
is a benchmark comparison when running a global max reduction on a 1-D Taichi field of 8M floats on an Nvidia GeForce RTX 3090 card:

* TLS disabled: 5.2 x 1e3 us
* TLS enabled: 5.7 x 1e1 us

TLS has led to an approximately 100x speedup.

## Block Local Storage (BLS)

Context: For a sparse field whose last layer is a `dense` SNode (i.e., its layer hierarchy matches `ti.root.(sparse SNode)+.dense`),
Taichi will assign one CUDA thread block to each `dense` container (or `dense` block). BLS optimization works specificially for such kinds of fields.

BLS aims to accelerate the stencil computation patterns by leveraging the CUDA shared memory. This optimization starts
with the users annotating the set of fields they would like to cache via `ti.block_local`. Taichi then attempts to
figure out the accessing range w.r.t the `dense` block of these annotated fields at *compile time*. If succeeded,
Taichi generates code that first fetches all the accessed data in range into a *block local* buffer (CUDA's shared memory),
then substitutes all the accesses to the corresponding slots into this buffer.

Here is an example illustrating the usage of BLS. `a` is a sparse field with a block size of `4x4`.

```python
a = ti.field(ti.f32)
b = ti.field(ti.f32)
# `a` has a block size of 4x4
ti.root.pointer(ti.ij, 32).dense(ti.ij, 4).place(a)

@ti.kernel
def foo():
  # Taichi will try to buffer `a` in CUDA shared memory
  ti.block_local(a)
  for i, j in a:
    print(a[i - 1, j], a[i, j + 2])
```

Each loop iteration accesses items with an offset `[-1, 0]` and `[0, 2]` to its coordinate, respectively.
Therefore, for an entire block spanning from `[M, N]` (inclusive) to `[M + 4, N + 4]` (exclusive), the accessed range w.r.t
this block is `[M - 1, M + 4) x [N, N + 6)` (derived from `[M + (-1), M + 4) x [N, N + 4 + 2)`). The mapping between the global coordinates `i, j` and the local indices
into the buffer is shown below:

![](../static/assets/bls_indices_mapping.png)

From a user's perspective, you do not need to worry about these underlying details. Taichi does all the inference and the global/block-local mapping
automatically. That is, Taichi will pre-allocate a CUDA shared memory buffer of size `5x6`, pre-load `a`'s data into this buffer, and
replace all the accesses to `a` (in the global memory) with the buffer in the loop body. While this simple example does not
modify `a`, if a block-cached field does get written, Taichi would also generate code that writes the buffer back to the global
memory.

:::note
BLS is not without its cost. Remember that BLS is designed for the stencil computation, where there is a large amount of
overlapped access to the global memory. If this is not the case, the pre-loading/post-storing could actually hurt the performance.
On top of that, recent generations of Nvidia's GPU cards have been closing the gap on the read-only access between the global memory
and the shared memory. Currently, we found BLS to be more effective for caching the destinations of the atomic operations. 

As a rule of thumb, run benchmarks to decide whether to enable BLS or not.
:::
