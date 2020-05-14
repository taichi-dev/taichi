Global settings
---------------

- Restart the Taichi runtime system (clear memory, destroy all variables and kernels): ``ti.reset()``
- Eliminate verbose outputs: ``ti.get_runtime().set_verbose(False)``
- To not trigger GDB when crashes: ``export TI_GDB_TRIGGER=0``
- To not use unified memory for CUDA: ``export TI_USE_UNIFIED_MEMORY=0``
- To specify pre-allocated memory size for CUDA: ``export TI_DEVICE_MEMORY_GB=0.5``
- Show more detailed log (TI_TRACE): ``export TI_LOG_LEVEL=trace``
- To specify which GPU to use for CUDA: ``export CUDA_VISIBLE_DEVICES=0``
- To specify which Arch to use: ``export TI_ARCH=cuda``
- To print intermediate IR generated: ``export TI_PRINT_IR=1``
- To print verbose details: ``export TI_VERBOSE=1``
