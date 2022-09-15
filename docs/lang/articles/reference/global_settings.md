---
sidebar_position: 3
---

# Global Settings


The call `ti.init()` is indispensable in every Taichi program. It allows you to customize your Taichi runtime by passing arguments to it or setting the corresponding environment variables. Each argument or environment variable will control one specific behavior of the Taichi runtime. We have introduced this function in our very first article ["getting started"](../get-started/index.md), but what given there was only a coarse sketch. In this article we will show its full functionalities.

In summary, when calling `ti.init()`, Taichi will execute the following steps to initialize a specific configuration: (we use the `arch` argument to illustrate this)

1. Read the arugment for that configuration that passed to it. In this case if Taichi reads `ti.init(arch=cuda)`, then it will use CUDA as the backend.
2. If that argument is missing, then Taichi will try to find if the corresponing environment variable is set. In this case if `export TI_ARCH=cuda` is set then it will also use CUDA as the backend.
3. If neither 1 or 2, Taichi will use a default configuration. In this case Taichi will use `arch=ti.cpu` as the default backend.


|     |     |     |     |     |
|:---:|:---:|:---:|:---:|:---:|
| Behavior    | Category    | init argument  | Environment variable    | Note |
|choose a backend |  Backends   |  `ti.init(arch=xxx)`   | `export TI_ARCH=xxx`    |  e.g. `export TI_ARCH=cuda`   |
| disable a backend on start up   |  Backends  |     |  `export TI_ENABLE_xxx=0`   |  e.g.  `export TI_ENABLE_opengl=0`  |
|  Choose GPU device for CUDA  |   Backends   |     | `export CUDA_VISIBLE_DEVICES=[gpuid]`   |  In case you want to use the CUDA backend together with GGUI on a machine with multiple GPU cards, please make sure `CUDA_VISIBLE_DEVICES` matches `TI_VISIBLE_DEVICE` if any of them exists. In general, `CUDA_VISIBLE_DEVICES` and `TI_VISIBLE_DEVICE` should point to a GPU device with the same UUID. Use `nvidia-smi -L` to query the details of your GPU devices |
| Choose GPU device for VULKAN    |   Backends   |     |   `export TI_VISIBLE_DEVICE=[gpuid]`  |     |
| specify pre-allocated CUDA memory size | Backends     |  `ti.init(device_memory_GB=0.5)`   |     |     |
| Disable advanced optimizations    |  Compilation   |  `ti.init(advanced_optimization=False)`   |     | This is for saing compile time and possible errors    |
| Disable fast math    |  Compilation   | `ti.init(fast_math=False)`   |     |  For preventing possible undefined math behavior   |
| print generated intermediate IR    |  Compilation   | `ti.init(print_ir=True)`    |     | Compiled kernels are [cached by default](https://docs.taichi-lang.org/docs/performance#offline-cache). To force compilation and IR emission, use `ti.init(print_ir=True, offline_cache=False)`    |
| start program in debug mode    | Runtime    |  `ti.init(debug=True)`   | `export TI_DEBUG=1`    | An equivalent way is to call your script via `ti debug your_script.py`  |
| disable importing torch on start up    | Runtime    |     |  `export TI_ENABLE_TORCH=0`   |     |
| disable importing paddle on start up    | Runtime    |     |   `export TI_ENABLE_PADDLE=0`  |     |
|set random seed for the random number generator | Runtime    | `ti.init(random_seed=seed)`    |     |   `ti.init(random_seed=int(time.time()))`  |
| Customize logging level    | Logging    |  `ti.init(log_level=ti.TRACE)`   | `export TI_LOG_LEVEL=trace`    | Equivalent to `ti.set_logging_level(ti.TRACE)`    |
| Eliminate verbose outputs    | Logging    | `ti.init(verbose=False)`    |     |     |
|trigger GDB when Taichi crashes    |  Develop   | `ti.init(gdb_trigger=True)`     |       |       |
|Cache compiled runtime bitcode in **dev mode** to save start up time  |  Develop   |      |   `export TI_CACHE_RUNTIME_BITCODE=1`    |       |
| specify how many threads to run test    |  Develop   |      |   `export TI_TEST_THREADS=4`    |  Equivalent to  `python tests/run_tests.py -t4`   |



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
