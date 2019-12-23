Global Settings
------------------

- Restart the Taichi runtime system (clear memory, destroy all variables and kernels): ``ti.reset()``
- Eliminate verbose outputs: ``ti.get_runtime().set_verbose(False)``
- To specify which GPU to use: ``export CUDA_VISIBLE_DEVICES=0``