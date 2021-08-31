from taichi.core.util import ti_core as _ti_core

# Kernel Profiler Mode
kernel_profiler_disable = _ti_core.KernelProfilerMode.disable
kernel_profiler_enable = _ti_core.KernelProfilerMode.enable
# CUDA backend
cuda_event = _ti_core.KernelProfilerMode.enable
cuda_accurate = _ti_core.KernelProfilerMode.cuda_accurate
cuda_detailed = _ti_core.KernelProfilerMode.cuda_detailed

kernel_profiler_modes = [
    True, False, kernel_profiler_disable, kernel_profiler_enable, cuda_event,
    cuda_accurate, cuda_detailed
]
kernel_profiler_mode_ids = [id(t) for t in kernel_profiler_modes]

__all__ = [
    'kernel_profiler_disable',
    'kernel_profiler_enable',
    'cuda_event',
    'cuda_accurate',
    'cuda_detailed',
    'kernel_profiler_modes',
]
