from microbenchmarks._items import BenchmarkItem
from microbenchmarks._utils import End2EndTimer, get_ti_arch

import taichi as ti


def end2end_executor(repeat, func, *args):
    # compile & warmup
    for i in range(repeat):
        func(*args)

    timer = End2EndTimer()
    timer.tick()
    for i in range(repeat):
        func(*args)
    time_in_s = timer.tock()
    return time_in_s * 1000 / repeat  #ms


def kernel_executor(repeat, func, *args):
    # compile & warmup
    for i in range(repeat):
        func(*args)
    ti.clear_kernel_profile_info()
    for i in range(repeat):
        func(*args)
    return ti.kernel_profiler_total_time() * 1000 / repeat  #ms


class MetricType(BenchmarkItem):
    name = 'get_metric'

    def __init__(self):
        self._items = {
            'kernel_elapsed_time_ms': kernel_executor,
            'end2end_time_ms': end2end_executor
        }

    @staticmethod
    def init_taichi(arch: str, metric_tag: str):
        if metric_tag == 'kernel_elapsed_time_ms':
            ti.init(kernel_profiler=True, arch=get_ti_arch(arch))
        elif metric_tag == 'end2end_time_ms':
            ti.init(kernel_profiler=False, arch=get_ti_arch(arch))
        else:
            return False
        return True
