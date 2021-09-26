---
sidebar_position: 4
---

# Profiler

## Overview
High-performance numerical computations is one of the design priorities of Taichi. We provide a series of profilers to quantify the performance of Taichi programs, help analyze where the bottleneck occurs, thus facilitate users optimize their code. These profilers are also designed as performance debugging tools for developers. 

The fellowing are profiling tools Taichi provides now:

    `ScopedProfiler` can be used to analyze the performance of Taichi compiler.

    `KernelProfiler` can be used to analyze the performance of Taichi kernels.

## KernelProfiler

The Trace Viewer shows you a timeline of the different events that occured on the CPU and the GPU during the profiling period.
To understand where the performance bottleneck occurs in the input pipeline
Time moves from left to right

For people from CUDA, Taichi kernels are similar to `__global__` functions.
1.  `KernelProfiler` records the costs of Taichi kernels on devices. To
    enable this profiler, set `kernel_profiler=True` in `ti.init`.
2.  Call `ti.print_kernel_profile_info()` to show the kernel profiling
    result. For example:

```python {3,13}
import taichi as ti

ti.init(ti.cpu, kernel_profiler=True)
var = ti.field(ti.f32, shape=1)


@ti.kernel
def compute():
    var[0] = 1.0


compute()
ti.print_kernel_profile_info()
```

The outputs would be:

```
[ 22.73%] jit_evaluator_0_kernel_0_serial             min   0.001 ms   avg   0.001 ms   max   0.001 ms   total   0.000 s [      1x]
[  0.00%] jit_evaluator_1_kernel_1_serial             min   0.000 ms   avg   0.000 ms   max   0.000 ms   total   0.000 s [      1x]
[ 77.27%] compute_c4_0_kernel_2_serial                min   0.004 ms   avg   0.004 ms   max   0.004 ms   total   0.000 s [      1x]
```

:::note
Currently the result of `KernelProfiler` could be incorrect on OpenGL
backend due to its lack of support for `ti.sync()`.
:::


CUPTI
In order to recover missed information, users needed to combine multiple tools together or manually add minimum correlation information to make sense of the data
capture detailed GPU hardware-level information and cannot 

capture GPU kernel events with high fidelity

## ScopedProfiler

1.  `ScopedProfiler` measures time spent on the **host tasks**
    hierarchically.
2.  This profiler is automatically on. To show its results, call
    `ti.print_profile_info()`. For example:

```python
import taichi as ti

ti.init(arch=ti.cpu)
var = ti.field(ti.f32, shape=1)



@ti.kernel
def compute():
    var[0] = 1.0
    print("Setting var[0] =", var[0])


compute()
ti.print_profile_info()
```

`ti.print_profile_info()` prints profiling results in a hierarchical format.

:::note
`ScopedProfiler` is a C++ class in the core of Taichi. It is not exposed
to Python users.
:::
