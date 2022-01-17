from microbenchmarks._items import BenchmarkItem
from microbenchmarks._utils import get_ti_arch

import taichi as ti


def kernel_executor(repeat, func, *args):
    # compile & warmup
    for i in range(repeat):
        func(*args)
    ti.clear_kernel_profile_info()
    for i in range(repeat):
        func(*args)
    return ti.kernel_profiler_total_time() * 1000 / repeat  #ms


class Result(BenchmarkItem):
    name = 'result'

    def __init__(self):
        self._items = {'kernel_elapsed_time_ms': {'impl': kernel_executor}}

    @staticmethod
    def init_taichi(arch: str, result_tag: str):
        if result_tag == 'kernel_elapsed_time_ms':
            ti.init(kernel_profiler=True, arch=get_ti_arch(arch))
        else:
            return False
        return True
