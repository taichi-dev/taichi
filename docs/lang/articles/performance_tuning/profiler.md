---
sidebar_position: 1
---

# Profiler

## Overview

High-performance numerical computation is one of the key application domains of Taichi.
We provide a suite of profilers to quantify the performance of Taichi programs, to help analyze where the bottleneck occurs,
and thus facilitate users optimizing their code. These profilers collect both hardware and Taichi-related information
and can also be used as performance debugging tools for developers.

Currently, Taichi provides the following profiling tools:
- `ScopedProfiler` is used to analyze the performance of the Taichi JIT compiler (host).
- `KernelProfiler` shows the performance of Taichi kernels (device), with detailed low-level performance metrics (such as memory bandwidth consumption) in its advanced mode.

## ScopedProfiler

`ScopedProfiler` tracks the time spent on **host tasks** such as JIT compilation.

1. This profiler is automatically on.
2. Call `ti.profiler.print_scoped_profiler_info()` to display results in a hierarchical format.

For example:

```python {13}
import taichi as ti

ti.init(arch=ti.cpu)
var = ti.field(ti.f32, shape=1)

@ti.kernel
def compute():
    var[0] = 1.0
    print("Setting var[0] =", var[0])

compute()
ti.profiler.print_scoped_profiler_info()
```

:::note
`ScopedProfiler` is a C++ class in Taichi.
:::


## KernelProfiler

`KernelProfiler` acquires the kernel profiling records from the backend, counts them in Python-scope, and prints the results to the console.

1. To enable this profiler, set `kernel_profiler=True` in `ti.init`.
2. To display the profiling results, call `ti.profiler.print_kernel_profiler_info()`. There are two modes of printing:
    - In `'count'` mode (default), the profiling records with the same kernel name are counted as a profiling result,
    and then the statistics are presented.
    - In `'trace'` mode, the profiler shows you a list of kernels that were launched on hardware during the profiling period.
    This mode provides more detailed performance information and runtime hardware metrics for each kernel.
3. To clear records in this profiler, call `ti.profiler.clear_kernel_profiler_info()`.

For example:

```python {3,13,14,18}
import taichi as ti

ti.init(ti.cpu, kernel_profiler=True)
x = ti.field(ti.f32, shape=1024*1024)

@ti.kernel
def fill():
    for i in x:
        x[i] = i

for i in range(8):
    fill()
ti.profiler.print_kernel_profiler_info('trace')
ti.profiler.clear_kernel_profiler_info()  # clear all records

for i in range(100):
    fill()
ti.profiler.print_kernel_profiler_info()  # default mode: 'count'
```

The outputs would be:

```
=========================================================================
X64 Profiler(trace)
=========================================================================
[      % |     time    ] Kernel name
[  0.00% |    0.000  ms] jit_evaluator_0_kernel_0_serial
[ 60.11% |    2.668  ms] fill_c4_0_kernel_1_range_for
[  6.06% |    0.269  ms] fill_c4_0_kernel_1_range_for
[  5.73% |    0.254  ms] fill_c4_0_kernel_1_range_for
[  5.68% |    0.252  ms] fill_c4_0_kernel_1_range_for
[  5.61% |    0.249  ms] fill_c4_0_kernel_1_range_for
[  5.63% |    0.250  ms] fill_c4_0_kernel_1_range_for
[  5.61% |    0.249  ms] fill_c4_0_kernel_1_range_for
[  5.59% |    0.248  ms] fill_c4_0_kernel_1_range_for
-------------------------------------------------------------------------
[100.00%] Total kernel execution time:   0.004 s   number of records:  9
=========================================================================
=========================================================================
X64 Profiler(count)
=========================================================================
[      %     total   count |      min       avg       max   ] Kernel name
[100.00%   0.033 s    100x |    0.244     0.329     2.970 ms] fill_c4_0_kernel_1_range_for
-------------------------------------------------------------------------
[100.00%] Total kernel execution time:   0.033 s   number of records:  1
=========================================================================
```

:::caution
Currently the result of `KernelProfiler` could be incorrect on OpenGL backend due to its lack of support for `ti.sync()`.
:::

### Advanced mode

For the CUDA backend, `KernelProfiler` has an experimental GPU profiling toolkit based on the Nvidia CUPTI, which has low and deterministic profiling overhead, and is able to capture more than 6000 hardware metrics.

Prerequisites to using CUPTI:
1. Install CUDA Toolkit.
2. Build Taichi from source with CUDA toolkit:
    - `TAICHI_CMAKE_ARGS="-DTI_WITH_CUDA_TOOLKIT:BOOL=ON" python3 setup.py develop --user`
3. Resolve privileges issue of Nvidia profiling module (run with `sudo` to get administrative privileges):
    - Add `options nvidia NVreg_RestrictProfilingToAdminUsers=0` to `/etc/modprobe.d/nvidia-kernel-common.conf`
    - Then `reboot` should resolve the permission issue (probably needs running `update-initramfs -u` before `reboot`)
    - See also [ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM).
