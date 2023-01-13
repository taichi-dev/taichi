---
sidebar_position: 3
---

# Global Settings

Every Taichi program starts with `ti.init()`. You can customize your Taichi runtime by passing arguments to `ti.init()` or setting environment variables. Each argument or environment variable controls one specific behavior of the Taichi runtime. For example, the argument `arch` specifies the backend, and the argument `debug` decides whether to run the program in debug mode.

Taichi executes the following to initialize a specific configuration in the `ti.init()` method call. Taking the `arch` argument as an example:

1. Taichi first looks for the arguments passed to `ti.init()`. In this case, after Taichi reads `ti.init(arch=cuda)`, it chooses CUDA as the backend and ignores the corresponding environment variable for backend setting, namely, `TI_ARCH`.
2. If no argument is found, Taichi checks the corresponding environment variable. In this case, if `arch` is not specified but the environment variable is set to `export TI_ARCH=cuda`, Taichi still chooses CUDA as the backend.
3. If no customized setting is found, Taichi uses a default configuration. In this case, if neither the argument `arch` is specified nor an environment variable `TI_ARCH` is found, Taichi adopts the default backend `arch=ti.cpu`.

Following are some frequently-used configurations in `ti.init()`:

**Customize Taichi runtime via arguments**

```
[Backend Options]

    arch: [ti.cpu, ti.gpu, ti.cuda, ti.vulkan, ...]
        Specify which architecture to use.

    device_memory_GB: float
        Specify the pre-allocated memory size for CUDA.


[Compilation Options]

    advanced_optimization: bool
        Enable/disable advanced optimization. Turning off the setting can save compile time and reduce possible errors.

    fast_math: bool
        Enable/disable fast math. Turning off the setting can prevent possible undefined math behavior.

    print_ir: bool
        Turn on/off the printing of the intermediate IR generated.


[Runtime Options]

    cpu_max_num_threads: int
        Set the number of threads used by the CPU thread pool.

    debug: bool
        Run your program in debug mode.

    default_cpu_block_dim: int
        Set the number of threads in a block on CPU.

    default_gpu_block_dim: int
        Set the number of threads in a block on GPU.

    default_fp: [ti.f32, ti.f64]
        Set the default precision of floating-point numbers in the Taichi scope.

    default_ip: [ti.i32, ti.i64]
        Set the default precision of integers in the Taichi scope.

    kernel_profiler: bool
        Turn on/off kernel performance profiling.

    offline_cache: bool
        Enable/disable offline cache of the compiled kernels.

    offline_cache_file_path: str
        Set a directory for holding the offline cached files.

    random_seed: int
        Set a custom seed for the random number generator.


[Logging Options]

    log_level: [ti.INFO, ti.TRACE, ti.WARN, ti.ERROR, ti.CRITICAL, ti.DEBUG]
        Set the logging level.

    verbose: bool
        Turn on/off verbose outputs. For example, `ti.init(verbose=False)` prevents verbose outputs.


[Develop Options]

    gdb_trigger: bool
        Enable/disable GDB when Taichi crashes. For example, `ti.init(gdb_trigger=True)` enables GDB.
```


**Customize Taichi runtime via environment variables**

Taichi allows a number of environment variables for runtime customization. Some provide an alternative to the `ti.init()` arguments listed above. For example, as mentioned above, `arch` is interchangeable with `TI_ARCH`. But not all of the environment variables can be replaced by arguments, and vice versa.

```
[Backend Options]

    CUDA_VISIBLE_DEVICES
        Specify which GPU to use for CUDA: `export CUDA_VISIBLE_DEVICES=[gpuid]`.

    TI_ARCH
        Specify which architecture to run the program. For example, `export TI_ARCH=cuda` designates CUDA as the backend.

    TI_ENABLE_[CUDA/OPENGL/...]
        Disable a backend upon startup. For example, set `export TI_ENABLE_CUDA=0` to disable the CUDA backend.

    TI_VISIBLE_DEVICE
        Specify which GPU to use for VULKAN: `export TI_VISIBLE_DEVICES=[gpuid]`.


[Runtime Options]

    TI_DEBUG
        Turn on/off debug mode. For example, `export TI_DEBUG=1` activates debug mode.

    TI_ENABLE_TORCH
        Enable/disable the import of torch upon startup. For example, `export TI_ENABLE_TORCH=0` prohibits the use of torch.
        The default value is 1.

    TI_ENABLE_PADDLE
        Enable/disable the import of paddle upon startup. For example, `export TI_ENABLE_PADDLE=0` prohibits the use of paddle.
        The default value is 1.


[Develop Options]

    TI_CACHE_RUNTIME_BITCODE
        Enable/disable the caching of compiled runtime bitcode in developer mode. For example, `export TI_CACHE_RUNTIME_BITCODE=1` enables the program to cache compiled runtime bitcode.
        Turning off the setting can save startup time.

    TI_TEST_THREADS
        Specify the number of threads to run a test. For example, set `export TI_TEST_THREADS=4` to allocate four threads.
        Alternatively, you can run `python tests/run_tests.py -t4`.


[Logging Options]

    TI_LOG_LEVEL
        Set the logging level. For example, `export TI_LOG_LEVEL=trace` enables the TRACE level.
```

## Backends

- To specify which architecture to use: `ti.init(arch=ti.cuda)` designates CUDA as the backend. This argument is equivalent to the environment variable `TI_ARCH`.
- To specify the pre-allocated memory size for CUDA: `ti.init(device_memory_GB=0.5)` allocates 0.5 GB size of memory.
- To specify which GPU to use for CUDA: `export CUDA_VISIBLE_DEVICES=[gpuid]`.
- To specify which GPU to use for VULKAN: `export TI_VISIBLE_DEVICE=[gpuid]`.
- To disable a backend (`CUDA`, `METAL`, `OPENGL`) upon startup: For example, `export TI_ENABLE_CUDA=0` disables CUDA.

:::note

If you want to use CUDA and Taichi's GGUI system at the same time on a machine with multiple GPU cards, ensure that `CUDA_VISIBLE_DEVICES` matches `TI_VISIBLE_DEVICE`. In principle, `CUDA_VISIBLE_DEVICES` and `TI_VISIBLE_DEVICE` should point to the same GPU device identified with UUID. Use `nvidia-smi -L` to retrieve the details of your GPU devices.

:::

## Compilation

- To disable advanced optimization: `ti.init(advanced_optimization=False)`, which helps save compile time and reduce possible errors.
- To disable fast math: `ti.init(fast_math=False)`, which helps prevent possible undefined math behavior.
- To print the intermediate IR generated: `ti.init(print_ir=True)`. Note that compiled kernels are [cached by default](../performance_tuning/performance.md#offline-cache). To force compilation and IR emission, use `ti.init(print_ir=True, offline_cache=False)`.


## Runtime

- To restart the entire Taichi system (and erase all fields and kernels): `ti.reset()`.
- To start a program in debug mode: `ti.init(debug=True)`. Alternatively, you can set the environment variable `TI_DEBUG` or run your code via `ti debug your_script.py`.
- To disable the import of torch upon startup: `export TI_ENABLE_TORCH=0`.
- To disable the import of paddle upon startup: `export TI_ENABLE_PADDLE=0`.
- To set a custom seed for the random number generator used by `ti.random()`: `ti.init(random_seed=seed)`. `seed` should be an integer. An example: `ti.init(random_seed=int(time.time()))`.
- To set the default precision of floating-point numbers of Taichi runtime to `ti.f64`: `ti.init(default_fp=ti.i64)`.
- To set the default precision of floating-point numbers of Taichi runtime to `ti.i32`: `ti.init(default_ip=ti.i32)`.
- To disable the offline cache of compiled kernels: `ti.init(offline_cache=False)`. See the [Offline cache](../performance_tuning/performance.md#offline-cache) for more information.
- To enable the use of variables as indices to access vector/matrix elements in the Taichi scope: `ti.init(dynamic_index=True)`.
- To turn on kernel profiling: `ti.init(kernel_profiler=True)`. See the [Profiler](../performance_tuning/profiler.md#kernelprofiler) for more information.


## Logging

- To set the logging level: `ti.init(log_level=ti.TRACE)` or `ti.set_logging_level(ti.TRACE)` enables the TRACE level. The environment variable `TI_LOG_LEVEL` serves the same purpose.
- To eliminate verbose outputs: `ti.init(verbose=False)`.

## Develop

- To trigger GDB when Taichi crashes: `ti.init(gdb_trigger=True)`.
- To cache compiled runtime bitcode in **dev mode**: `export TI_CACHE_RUNTIME_BITCODE=1`, which saves startup time.
- To allocate four threads to run a test: `export TI_TEST_THREADS=4` or `python tests/run_tests.py -t4`.


:::note

If `ti.init` is called twice, the configuration in the first call is discarded. For example:

```python {1,3}
ti.init(debug=True)
print(ti.cfg.debug)  # True
ti.init()
print(ti.cfg.debug)  # False
```

:::


## Going high precision

The default setting `fast_math=True` may cause unexpected precision problems, which can be hard to debug. For example, the function `ti.sqrt` (and those that rely on it, such as `ti.norm`) may have a lower precision on the CUDA backend than on the CPU backends. See this [issue](https://github.com/taichi-dev/taichi/issues/6346) for more information. To work around this problem, set the default precision to `ti.f64` and turn off the `fast_math` option:

```python
ti.init(default_fp=ti.f64, fast_math=False)
```

If you encounter other precision loss problems, set `ti.init(fast_math=False)` and see if it works. If not, [report an issue](https://github.com/taichi-dev/taichi/issues) to us.
