---
sidebar_position: 5
---

# Performance tuning

## For-loop decorators

In Taichi kernels, for-loop in the outermost scope is automatically
parallelized. Our compiler automatically tunes the parameters to best explore
the target architecture. Nevertheless, for Ninjas who strive for the last few %
of performance, we also provide some APIs to allow developers fine-tune their
applications. For example, specifying a suitable `ti.block_dim` could yield an almost
3x performance boost in
[examples/mpm3d.py](https://github.com/taichi-dev/taichi/blob/master/examples/mpm3d.py).

:::note
For **performance profiling** utilities, please see [Profiler section of the Contribution Guide](../misc/profiler.md).
:::

### Background: Thread hierarchy of GPUs

To better understand how the mentioned for-loop is parallelized, we briefly
introduce the **thread hierarchy** on modern GPU architectures.

From a fine-grained to a coarse-grained level, the computation units can be
defined as: **iteration** \< **thread** \< **block** \< **grid**.

- **iteration**: Iteration is the **body of a for-loop**. Each
  iteration corresponding to a specific `i` value in for-loop.
- **thread**: Iterations are grouped into threads. A thread is the
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
Note that we employ the CUDA terminology here, other backends such as OpenGL and Metal follow a similar thread hierarchy.

### Example: Tuning the block-level parallelism of a for-loop

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
