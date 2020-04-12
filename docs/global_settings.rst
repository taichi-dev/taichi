Global Settings
------------------

- Restart the Taichi runtime system (clear memory, destroy all variables and kernels): ``ti.reset()``
- Eliminate verbose outputs: ``ti.get_runtime().set_verbose(False)``
- Not to trigger GDB when crashes: ``ti.set_gdb_trigger(False)``
- Show more detailed log (TI_TRACE): ``export TI_LOG_LEVEL=trace``
- To specify which GPU to use for CUDA: ``export CUDA_VISIBLE_DEVICES=0``
- To specify which Arch to use: ``export TI_ARCH=cuda``
- To print intermediate IR generated: ``export TI_PRINT_IR=1``
