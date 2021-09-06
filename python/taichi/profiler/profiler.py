from taichi.core import ti_core as _ti_core


class Profiler:

    # toolkit
    # default = "default"  #COMMON [True, False]     ──┐
    #                                                  ├── equivalent
    # cuevent = "cuevent"  #CUDA   [disable, enable] ──┘
    # cupti   = "cupti"    #CUPTI  [onepass, detailed, customized]

    # kernel_profiler_mode_ids = [id(t) for t in kernel_profiler_modes]

    # kernel profiler common
    disable = _ti_core.KernelProfilingMode.disable
    enable = _ti_core.KernelProfilingMode.enable

    # kernel profiler CUDA backend
    cuda_event = _ti_core.KernelProfilingMode.enable
    cupti_onepass = _ti_core.KernelProfilingMode.cupti_onepass
    cupti_detailed = _ti_core.KernelProfilingMode.cupti_detailed
    cupti_customized = _ti_core.KernelProfilingMode.cupti_customized

    kernel_profiler_modes = [
        True, False, disable, enable, cuda_event, cupti_onepass, cupti_detailed
    ]

    mode_ = disable

    def __init__(self, mode=None):
        self.mode_ = self.get_kernel_profiler_mode(mode)

    def get_kernel_profiler_mode(self, kernel_profiler=None):
        if kernel_profiler is None:
            return self.disable
        if kernel_profiler is False:
            return self.disable
        elif kernel_profiler is True:
            return self.enable
        elif type(kernel_profiler) == _ti_core.KernelProfilingMode:
            return kernel_profiler
        else:
            _ti_core.warn(
                f'kernel_profiler mode error : {type(kernel_profiler)}')
