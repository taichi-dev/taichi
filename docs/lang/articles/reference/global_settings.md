---
sidebar_position: 3
---

# Global Settings


The call `ti.init()` is indispensable in every Taichi program. It allows you to customize your Taichi runtime by passing arguments to it or setting the corresponding environment variables. Each argument or environment variable will control one specific behavior of the Taichi runtime. We have introduced this function in our very first article ["getting started"](../get-started/index.md), but what given there was only a coarse sketch. In this article we will show its full functionalities.

In summary, when calling `ti.init()`, Taichi will execute the following steps to initialize a specific configuration: (we use the `arch` argument to illustrate this)

1. Firstly Taichi will try to find if the corresponing environment variable is set. In this case if `export TI_ARCH=cuda` is set then it will use CUDA as the backend and omit the argument `arch` passed to `ti.init()`.
2. If the environment variable is not set then Taichi will read the arugment for that configuration that passed to `ti.init()`. In this case if Taichi reads `ti.init(arch=cuda) then it will choose CUDA as the backend.
3. If neither 1 or 2, Taichi will use a default configuration. In this case Taichi will use `arch=ti.cpu` as the default backend.


|     |     |     |     |     |
|:---:|:---:|:---:|:---:|:---:|
| Behavior    | Category    | `ti.init()` argument  | Environment variable    | Note |
|choose a backend |  Backends   |  `arch=xxx`   | `export TI_ARCH=xxx`    |  e.g. `export TI_ARCH=cuda`   |
| disable a backend on start up   |  Backends  |     |  `export TI_ENABLE_xxx=0`   |  e.g.  `export TI_ENABLE_opengl=0`  |
|  Choose CUDA device |   Backends   |     | `export CUDA_VISIBLE_DEVICES=[gpuid]`   |   |
| Choose Vulkan device   |   Backends   |     |   `export TI_VISIBLE_DEVICE=[gpuid]`  |     |
| Specify pre-allocated CUDA memory size | Backends     |  `device_memory_GB=0.5`   |     |     |
| Disable advanced optimizations    |  Compilation   |  `advanced_optimization=False`   |     | This is for saing compile time and possible errors    |
| Disable fast math    |  Compilation   | `fast_math=False`   |     |  For preventing possible undefined math behavior   |
| Print generated intermediate IR    |  Compilation   | `print_ir=True`    |     | Compiled kernels are [cached by default](https://docs.taichi-lang.org/docs/performance#offline-cache). To force compilation and IR emission, use `ti.init(print_ir=True, offline_cache=False)`    |
| Start program in debug mode    | Runtime    |  `debug=True`   | `export TI_DEBUG=1`    | An equivalent way is to call your script via `ti debug your_script.py`  |
| Disable importing torch on start up    | Runtime    |     |  `export TI_ENABLE_TORCH=0`   |     |
| Disable importing paddle on start up    | Runtime    |     |   `export TI_ENABLE_PADDLE=0`  |     |
|Set random seed | Runtime    | `random_seed=seed`    |     | `seed` is an integer |
| Customize logging level    | Logging    |  `log_level=ti.TRACE`   | `export TI_LOG_LEVEL=trace`    | Equivalent to `ti.set_logging_level(ti.TRACE)`    |
| Eliminate verbose outputs    | Logging    | `verbose=False`    |     |     |
|trigger GDB when Taichi crashes    |  Develop   | `gdb_trigger=True`     |       |       |
|Cache compiled runtime bitcode in **dev mode** |  Develop   |      |   `export TI_CACHE_RUNTIME_BITCODE=1`    |   To save start up time      |
| Specify how many threads to run test    |  Develop   |      |   `export TI_TEST_THREADS=4`    |  Equivalent to  `python tests/run_tests.py -t4`   |



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