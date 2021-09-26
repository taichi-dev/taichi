---
sidebar_position: 4
---

# Profiler

## Overview
High-performance numerical computation is one of the design priorities of Taichi. We provide a series of profilers to quantify the performance of Taichi programs, help analyze where the bottleneck occurs, and thus facilitate users optimizing their code. These profilers collect both hardware and Taichi-related information and can also be used as performance debugging tools for developers.

The fellows are profiling tools Taichi provides now:
- `ScopedProfiler` can be used to analyze the performance of the Taichi compiler.
- `KernelProfiler` can be used to analyze the performance of Taichi kernels. #TODO

## ScopedProfiler
`ScopedProfiler` measures time spent on the **host tasks**.

1. This profiler is automatically on. 
2. call `ti.print_profile_info()`. To show its hierarchical formatted results. 

For example:

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

:::note
`ScopedProfiler` is a C++ class in the core of Taichi. It is not exposed to Python users.
:::


## KernelProfiler

`KernelProfiler` acquires kernel profiling records from backend, counts them in python-scope, and outputs results by printing.

1. To enable this profiler, set `kernel_profiler=True` in `ti.init`.
2. Call `ti.print_kernel_profile_info()` to show the kernel profiling result, there are two modes to print:
    - In `'count'` mode (default mode), records with the same kernel name are counted as a profiling result, and then presented in a statistical perspective.
    - The `'trace'` mode shows you a table of kernels that launched on hardware (e.g. CPU,GPU) during the profiling period. This mode provides more hardware performance information for each kernel.
3. Use `ti.clear_kernel_profile_info()` to clear records in this profiler.

For example:
```python {3,13}
import taichi as ti

ti.init(ti.cpu, kernel_profiler=True)
x = ti.field(ti.f32, shape=1024*1024)

@ti.kernel
def fill():
    for i in x:
        x[i] = i

for i in range(8):
    fill()
ti.print_kernel_profile_info('trace')
ti.clear_kernel_profile_info() #clear

for i in range(100):
    fill()
ti.print_kernel_profile_info() #default mode: 'count'
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

:::note
Currently the result of `KernelProfiler` could be incorrect on OpenGL backend due to its lack of support for `ti.sync()`.
:::

### Advanced 
For CUDA backend, `KernelProfiler` has a experimental GPU profiling toolkit, Nvidia CUPTI APIs, and is able to capture ...

Prerequisites to use CUPTI:
1. Install CUDA Toolkit
2. Add environment variables :
    - `export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda` to your shell configuration files such as `~/.bashrc` and `~/.zshrc`.
3. Build Taichi from source with CUDA toolkit: 
    - `TAICHI_CMAKE_ARGS="-DTI_WITH_CUDA_TOOLKIT:BOOL=ON" python3 setup.py develop --user`.
4. Resolve privileges issue of Nvidia profiling module (Ubuntu 20.04): 
    - Add `options nvidia NVreg_RestrictProfilingToAdminUsers=0` to `/etc/modprobe.d/nvidia-kernel-common.conf`, 
    - then `reboot` should resolve the permision issue (Probably needs running `update-initramfs -u` before `reboot`).