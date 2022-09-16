---
sidebar_position: 3
---

# Global Settings


The call `ti.init()` is indispensable in every Taichi program. It allows you to customize your Taichi runtime by passing arguments to it or setting the corresponding environment variables. Each argument or environment variable will control one specific behavior of the Taichi runtime. We have introduced this function in our very first article ["getting started"](../get-started/index.md), but what given there was only a coarse sketch. In this article we will show its full functionalities.

In summary, when calling `ti.init()`, Taichi will execute the following steps to initialize a specific configuration: (we use the `arch` argument to illustrate this)

1. Firstly Taichi will read the argument for that configuration that is passed to `ti.init()`. In this case if Taichi finds `ti.init(arch=cuda)` then it will choose CUDA as the backend and omit the environment variable.
2. If the argument is not found then Taichi will check if the corresponding environment variable is set. In this case if `arch` is not specified but `export TI_ARCH=cuda` is set then Taichi will use CUDA as the backend.
3. If neither 1 or 2, Taichi will use a default configuration. In this case if neither the argument `arch` is specified nor the environment variable `TI_ARCH` is found, Taichi will use `arch=ti.cpu` as the default backend.


Below are some most frequently used configurations you can set with the `ti.init()` call:

**Customize via `ti.init()`**

```
[Backend Options]

    arch: [ti.cpu, ti.gpu, ti.cuda, ti.vulkan, ...]
        Specify which architecture to use.
        The corresponding environment variable is `TI_ARCH`.

    device_memory_GB: float
        Specify the pre-allocated memory size for CUDA.
        For example `ti.init(device_memory_GB=0.5)` will allocate 0.5GB memory size.

[Compilation Options]

    advanced_optimization: bool
        Enable/disable advanced optimization to save compile time & possible errors.

    fast_math: bool
        Enable/disable fast math to prevent possible undefined math behavior.

    print_ir: bool
        Turn on/off printing intermediate IR generated.

[Runtime Options]

    cpu_max_num_threads: int
        Set the number of threads used by the CPU thread pool.

    debug: bool
        Run program in debug mode. Equivalently you can run your code via `ti debug your_script.py`.
        The corresponding environment variable is `TI_DEBUG`.

    default_cpu_block_dim: int
        Set the number of threads in a block on CPU.

    default_gpu_block_dim: int
        Set the number of threads in a block on GPU.

    default_fp: [ti.f32, ti.f64]
        Set the default precision for floating-point numbers in the Taichi scope.

    default_io: [ti.i32, ti.i64]
        Set the default precision for integers in the Taichi scope.

    dynamic_index: bool
        Enable/disable vector/matrix indexing using variables.

    kernel_profiler: bool
        Turn on/off kernel performance profiling.

    offline_cache: bool
        Enable/disable offline cache of the compiled kernels.

    offline_cache_file_path: str
        Directory holding the offline cached files.

    packed: bool
        Enable/disable the packed memory layout. See https://docs.taichi-lang.org/docs/layout.

    random_seed: int
        Set a custom seed for the random number generator. e.g. `ti.init(random_seed=1)`.

[Logging Options]

    log_level: [ti.INFO, ti.TRACE, ti.WARN, ti.ERROR, ti.CRITICAL, ti.DEBUG]
        Set the logging level. e.g. `ti.init(log_level=ti.TRACE)`.
        The corresponding environment variable is `TI_LOG_LEVEL`.

    verbose: bool
        Eliminate verbose outputs. e.g. `ti.init(verbose=False)`.

[Develop Options]

    gdb_trigger: bool
        To trigger GDB when Taichi crashes. e.g. `ti.init(gdb_trigger=True)`.
```


**Customize via environment variables**

Below are some environment variables that you can set to customize your Taichi program, they overlap but are not in one-to-one correspondence with the `ti.init()` arguments listed above:

```
[Backend Options]

    CUDA_VISIBLE_DEVICES
        Specify which GPU to use for CUDA: `export CUDA_VISIBLE_DEVICES=[gpuid]`.

    TI_ARCH
        Specify which architecture to run the program, e.g. `export TI_ARCH=cuda`.

    TI_ENABLE_[CUDA/OPENGL/...]
        Disable a backend on start up, e.g. CUDA: `export TI_ENABLE_CUDA=0`.

    TI_VISIBLE_DEVICE
        Specify which GPU to use for VULKAN: `export TI_VISIBLE_DEVICES=[gpuid]`.
        
        
[Runtime Options]

    TI_DEBUG
        Turn on/off the debug mode. e.g. `export TI_DEBUG=1`.

    TI_ENABLE_TORCH
        Enable/disable importing torch on start up, e.g. `export TI_ENABLE_TORCH=0` to disable.
        The default is 1.
    
    TI_ENABLE_PADDLE
        Enable/disable importing paddle on start u, e.g. `export TI_ENABLE_PADDLE=0` to disable.
        The default is 1.
        

[Develop Options]
 
    TI_CACHE_RUNTIME_BITCODE
        Enable/disable caching compiled runtime bitcode in dev mode to save start up time, e.g. `export TI_CACHE_RUNTIME_BITCODE=1` to enable.
        
    TI_TEST_THREADS
        Specify how many threads to run test, e.g. `export TI_TEST_THREADS=4`.
        Equivlently you can run `python tests/run_tests.py -t4`.
        
[Logging Options]

    TI_LOG_LEVEL
        Set the logging level, e.g. `export TI_LOG_LEVEL=trace`.
```


:::note

If `ti.init` is called twice, then the configuration in first invocation
will be completely discarded, e.g.:

```python {1,3}
ti.init(debug=True)
print(ti.cfg.debug)  # True
ti.init()
print(ti.cfg.debug)  # False
```

:::

:::note

In case you want to use taichi cuda backend together with GGUI on a machine with multiple GPU cards, please make sure `CUDA_VISIBLE_DEVICES` matches `TI_VISIBLE_DEVICE` if any of them exists. In general, `CUDA_VISIBLE_DEVICES` and `TI_VISIBLE_DEVICE` should point to a GPU device with the same UUID. Use `nvidia-smi -L` to query the details of your GPU devices.
:::
