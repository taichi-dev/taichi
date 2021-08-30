from taichi.core.util import ti_core as _ti_core

# Kernel Profiler Mode
kernel_profiler_disable = _ti_core.KernelProfilerMode.disable
kernel_profiler_enable  = _ti_core.KernelProfilerMode.enable
cuda_accurate           = _ti_core.KernelProfilerMode.cuda_accurate
cuda_detailed           = _ti_core.KernelProfilerMode.cuda_detailed

kernel_profiler_enums = [kernel_profiler_disable, kernel_profiler_enable, cuda_accurate, cuda_detailed]
kernel_profiler_enums_ids = [id(t) for t in kernel_profiler_enums]

__all__ = [
    'kernel_profiler_disable',
    'kernel_profiler_enable',
    'cuda_accurate',
    'cuda_detailed',
]
