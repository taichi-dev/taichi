Global Settings
------------------

- Restart the Taichi runtime system (clear memory, destroy all variables and kernels): ``ti.reset()``
- Eliminate verbose outputs: ``ti.get_runtime().set_verbose(False)``
- Not to trigger GDB when crashes: ``ti.misc.util.set_gdb_trigger(False)``
- Show more detailed log (TI_TRACE): ``ti.misc.util.set_logging_level(ti.TRACE)``
- To specify which GPU to use for CUDA: ``export CUDA_VISIBLE_DEVICES=0``
- To specify which Arch to use: ``export TI_ARCH=cuda``
