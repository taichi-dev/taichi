Global settings
---------------

Backends
********

- To specify which Arch to use: ``export TI_ARCH=cuda`` or ``ti.init(arch=ti.cuda)``.
- To specify which GPU to use for CUDA: ``export CUDA_VISIBLE_DEVICES=[gpuid]``.
- To specify pre-allocated memory size for CUDA: ``export TI_DEVICE_MEMORY_GB=0.5`` or ``ti.init(device_memory_GB=0.5)``.
- To disable unified memory usage on CUDA: ``export TI_USE_UNIFIED_MEMORY=0``.

Compilation
***********

- Disable advanced optimization to save compile time & possible erros: ``ti.core.toggle_advanced_optimization(False)``.
- To print intermediate IR generated: ``export TI_PRINT_IR=1`` or ``ti.init(print_ir=True)``.
- To print preprocessed Python code: ``export TI_PRINT_PREPROCESSED=1`` or ``ti.init(print_preprocessed=True)``..

Runtime
*******

- Restart the entire Taichi system (destroy all tensors and kernels): ``ti.reset()``.
- To start program in debug mode: ``export TI_DEBUG=1`` or ``ti.init(debug=True)``.

Logging
*******

- Show more detailed log to level TRACE: ``export TI_LOG_LEVEL=trace`` or ``ti.set_logging_level(ti.TRACE)``.
- Eliminate verbose outputs: ``ti.get_runtime().set_verbose(False)`` or ``TI_VERBOSE=0``.

Develop
*******

- Cache compiled runtime bitcode in **dev mode** to save start up time: ``export TI_CACHE_RUNTIME_BITCODE=1``.
- To trigger GDB when Taichi crashes: ``export TI_GDB_TRIGGER=1``.
