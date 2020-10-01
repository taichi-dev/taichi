import taichi as ti
import functools
import os


def benchmark_async(func):
    @functools.wraps(func)
    def body():
        for arch in [ti.cpu, ti.cuda]:
            for async_mode in [True, False]:
                os.environ['TI_CURRENT_BENCHMARK'] = func.__name__
                ti.init(arch=arch, async_mode=async_mode, kernel_profiler=True)
                if arch == ti.cpu:
                    scale = 2
                else:
                    # Use more data to hide compilation overhead
                    # (since CUDA runs much faster than CPUs)
                    scale = 64
                func(scale)

    return body
