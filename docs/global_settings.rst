Global settings
---------------

Backends
********

- To specify which Arch to use: ``ti.init(arch=ti.cuda)``.
- To specify pre-allocated memory size for CUDA: ``ti.init(device_memory_GB=0.5)``.
- To disable unified memory usage on CUDA: ``ti.init(use_unified_memory=False)``.
- To specify which GPU to use for CUDA: ``export CUDA_VISIBLE_DEVICES=[gpuid]``.
- To disable a backend on start up, say, CUDA: ``export TI_ENABLE_CUDA=0``.

Compilation
***********

- Disable advanced optimization to save compile time & possible errors: ``ti.init(advanced_optimization=False)``.
- Disable fast math to prevent possible undefined math behavior: ``ti.init(fast_math=False)``.
- To print preprocessed Python code: ``ti.init(print_preprocessed=True)``.
- To show pretty Taichi-scope stack traceback: ``ti.init(excepthook=True)``.
- To print intermediate IR generated: ``ti.init(print_ir=True)``.

Runtime
*******

- Restart the entire Taichi system (destroy all fields and kernels): ``ti.reset()``.
- To start program in debug mode: ``ti.init(debug=True)`` or ``ti debug your_script.py``.
- To disable importing torch on start up: ``export TI_ENABLE_TORCH=0``.

Logging
*******

- Show more detailed log to level TRACE: ``ti.init(log_level=ti.TRACE)`` or ``ti.set_logging_level(ti.TRACE)``.
- Eliminate verbose outputs: ``ti.init(verbose=False)``.

Develop
*******

- To trigger GDB when Taichi crashes: ``ti.init(gdb_trigger=True)``.
- Cache compiled runtime bitcode in **dev mode** to save start up time: ``export TI_CACHE_RUNTIME_BITCODE=1``.
- To specify how many threads to run test: ``export TI_TEST_THREADS=4`` or ``ti test -t4``.


Specifying ``ti.init`` arguments from environment variables
***********************************************************

Arguments for ``ti.init`` may also be specified from environment variables. For example:

- ``ti.init(arch=ti.cuda)`` is equivalent to ``export TI_ARCH=cuda``.
- ``ti.init(log_level=ti.TRACE)`` is equivalent to ``export TI_LOG_LEVEL=trace``.
- ``ti.init(debug=True)`` is equivalent to ``export TI_DEBUG=1``.
- ``ti.init(use_unified_memory=False)`` is equivalent to ``export TI_USE_UNIFIED_MEMORY=0``.

If both ``ti.init`` argument and the corresponding environment variable are specified, then
the one in the environment variable will **override** the one in the argument, e.g.:

- if ``ti.init(arch=ti.cuda)`` and ``export TI_ARCH=opengl`` are specified at the same time,
  then Taichi will choose ``ti.opengl`` as backend.
- if ``ti.init(debug=True)`` and ``export TI_DEBUG=0`` are specified at the same time,
  then Taichi will disable debug mode.

.. note::

    If ``ti.init`` is called twice, then the configuation in first invocation will be
    completely discarded, e.g.:

    ::

        ti.init(debug=True)
        print(ti.cfg.debug)  # True
        ti.init()
        print(ti.cfg.debug)  # False
