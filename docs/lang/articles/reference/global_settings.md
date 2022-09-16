---
sidebar_position: 3
---

# Global Settings


The call `ti.init()` is indispensable in every Taichi program. It allows you to customize your Taichi runtime by passing arguments to it or setting the corresponding environment variables. Each argument or environment variable will control one specific behavior of the Taichi runtime. We have introduced this function in our very first article ["getting started"](../get-started/index.md), but what given there was only a coarse sketch. In this article we will show its full functionalities.

In summary, when calling `ti.init()`, Taichi will execute the following steps to initialize a specific configuration: (we use the `arch` argument to illustrate this)

1. Firstly Taichi will read the argument for that configuration that is passed to `ti.init()`. In this case if Taichi finds `ti.init(arch=cuda)` then it will choose CUDA as the backend and omit the environment variable.
2. If the argument is not found then Taichi will check if the corresponding environment variable is set. In this case if `arch` is not specified but `export TI_ARCH=cuda` is set then Taichi will use CUDA as the backend.
3. If neither 1 or 2, Taichi will use a default configuration. In this case if neither the argument `arch` is specified nor the environment variable `TI_ARCH` is found, Taichi will use `arch=ti.cpu` as the default backend.


Below are some most frequently used configurations:

```
advanced_optimization: bool
    Enable/disable advanced optimization to save compile time & possible errors.

arch: ti.cpu/ti.gpu/ti.cuda/...
    Specify which architecture (Arch) to use.

debug: bool
    Run program in debug mode.

device_memory_GB: float
    Specify the pre-allocated memory size for CUDA.

fast_math: bool
    Enable/disable fast math to prevent possible undefined math behavior.

log_level:
    Set the logging level.

print_ir: bool
    Turn on/off printing intermediate IR generated.

random_seed: int
    Set a custom seed for the random number generator.
```


## Backends


- To specify which GPU to use for CUDA:
  `export CUDA_VISIBLE_DEVICES=[gpuid]`.
- To specify which GPU to use for VULKAN:
- `export TI_VISIBLE_DEVICE=[gpuid]`
- To disable a backend (`CUDA`, `METAL`, `OPENGL`) on start up, e.g. CUDA:
  `export TI_ENABLE_CUDA=0`.

:::note

In case you want to use taichi cuda backend together with GGUI on a machine with multiple GPU cards, please make sure `CUDA_VISIBLE_DEVICES` matches `TI_VISIBLE_DEVICE` if any of them exists. In general, `CUDA_VISIBLE_DEVICES` and `TI_VISIBLE_DEVICE` should point to a GPU device with the same UUID. Use `nvidia-smi -L` to query the details of your GPU devices.

## Compilation

- To print intermediate IR generated: `ti.init(print_ir=True)`. Note that compiled kernels are [cached by default](https://docs.taichi-lang.org/docs/performance#offline-cache). To force compilation and IR emission, use `ti.init(print_ir=True, offline_cache=False)`.

## Runtime

- Restart the entire Taichi system (destroy all fields and kernels):
  `ti.reset()`.
- To start program in debug mode: `ti.init(debug=True)` or
  `ti debug your_script.py`.
- To disable importing torch on start up: `export TI_ENABLE_TORCH=0`.
- To disable importing paddle on start up: `export TI_ENABLE_PADDLE=0`.
- To set a custom seed for the random number generator used by `ti.random()`: `ti.init(random_seed=seed)` where `seed` is an integer. For example `ti.init(random_seed=int(time.time()))`.

## Logging

- Show more detailed log to level TRACE: `ti.init(log_level=ti.TRACE)`
  or `ti.set_logging_level(ti.TRACE)`.
- Eliminate verbose outputs: `ti.init(verbose=False)`.

## Develop

- To trigger GDB when Taichi crashes: `ti.init(gdb_trigger=True)`.
- Cache compiled runtime bitcode in **dev mode** to save start up
  time: `export TI_CACHE_RUNTIME_BITCODE=1`.
- To specify how many threads to run test: `export TI_TEST_THREADS=4`
  or `python tests/run_tests.py -t4`.

## Specifying `ti.init` arguments from environment variables

Arguments for `ti.init` may also be specified from environment
variables. For example:

- `ti.init(arch=ti.cuda)` is equivalent to `export TI_ARCH=cuda`.
- `ti.init(log_level=ti.TRACE)` is equivalent to
  `export TI_LOG_LEVEL=trace`.
- `ti.init(debug=True)` is equivalent to `export TI_DEBUG=1`.

If both `ti.init` argument and the corresponding environment variable
are specified, then the one in the environment variable will
**override** the one in the argument, e.g.:

- if `ti.init(arch=ti.cuda)` and `export TI_ARCH=opengl` are specified
  at the same time, then Taichi will choose `ti.opengl` as backend.
- if `ti.init(debug=True)` and `export TI_DEBUG=0` are specified at
  the same time, then Taichi will disable debug mode.

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
