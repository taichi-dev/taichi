---
sidebar_position: 3
---

# Global Settings


The call `ti.init()` is indispensable in every Taichi program. It allows you to customize your Taichi runtime by passing arguments to it or setting the corresponding environment variables. Each argument or environment variable will control one specific behavior of the Taichi runtime. We have introduced this function in our very first article ["getting started"](../get-started/index.md), but what given there was only a coarse sketch. In this article we will show its full functionalities.

In summary, when calling `ti.init()`, Taichi will execute the following steps to initialize a specific configuration: (we use the `arch` argument to illustrate this)

1. Firstly Taichi will try to find if the corresponing environment variable is set. In this case if `export TI_ARCH=cuda` is set then it will use CUDA as the backend and omit the argument `arch` passed to `ti.init()`.
2. If the environment variable is not set then Taichi will read the arugment for that configuration that passed to `ti.init()`. In this case if Taichi reads `ti.init(arch=cuda) then it will choose CUDA as the backend.
3. If neither 1 or 2, Taichi will use a default configuration. In this case Taichi will use `arch=ti.cpu` as the default backend.


Below are some most frequently used configurations, grouped by categories:


## Backends



| Behavior | Option  | Note |
|:---:|:---:|:---|
| Choose a backend | `arch=ti.cpu/gpu/cuda/...` or  `export TI_ARCH=cuda/opengl/...`| e.g. `export TI_ARCH=cuda`|
| Disable a backend on start up| `export TI_ENABLE_cuda/opengl/...=0`   |  e.g.  `export TI_ENABLE_opengl=0`  |
|  Choose CUDA device | `export CUDA_VISIBLE_DEVICES=[gpuid]`   |   |
| Choose Vulkan device |`export TI_VISIBLE_DEVICE=[gpuid]`|     |
| Specify pre-allocated CUDA memory size | `device_memory_GB=0.5`|   |

## Compilation

| Behavior | Option  | Note |
|:---:|:---:|:---|
| Disable advanced optimizations |`advanced_optimization=False`| This is for saving compile time and possible errors|
| Disable fast math |`fast_math=False`   |For preventing possible undefined math behavior   |
| Print generated intermediate IR |`print_ir=True`| Compiled kernels are [cached by default](https://docs.taichi-lang.org/docs/performance#offline-cache). To force compilation and IR emission, use `ti.init(print_ir=True, offline_cache=False)`|

## Runtime


| Behavior | Option  | Note |
|:---:|:---:|:---|
| Start program in debug mode    | `debug=True` or `export TI_DEBUG=1` | An equivalent way is to call your script via `ti debug your_script.py`  |
| Disable importing torch on start up    |`export TI_ENABLE_TORCH=0`   |     |
| Disable importing paddle on start up   |`export TI_ENABLE_PADDLE=0`  |     |
|Set random seed | Runtime    | `random_seed=seed`| `seed` is an integer |


## Logging

| Behavior | Option  | Note |
|:---:|:---:|:---|
| Customize logging level    |`log_level=ti.TRACE` or `export TI_LOG_LEVEL=trace`| Equivalent to `ti.set_logging_level(ti.TRACE)` |
| Eliminate verbose outputs    | `verbose=False`|     |

## Develop

| Behavior | Option  | Note |
|:---:|:---:|:---|
|Trigger GDB when Taichi crashes  | `gdb_trigger=True` |       |
|Cache compiled runtime bitcode in **dev mode** |   `export TI_CACHE_RUNTIME_BITCODE=1`    |   To save start up time |
| Specify how many threads to run test |   `export TI_TEST_THREADS=4`    |  Equivalent to  `python tests/run_tests.py -t4` |


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

In case you want to use the CUDA backend together with GGUI on a machine with multiple GPU cards, please make sure `CUDA_VISIBLE_DEVICES` matches `TI_VISIBLE_DEVICE` if any of them exists. In general, `CUDA_VISIBLE_DEVICES` and `TI_VISIBLE_DEVICE` should point to a GPU device with the same UUID. Use `nvidia-smi -L` to query the details of your GPU devices

:::
