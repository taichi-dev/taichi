import taichi as ti
import os
import sys

sys.path.append(os.path.join(ti.core.get_repo_dir(), 'tests', 'python'))

from fuse_test_template import template_fuse_dense_x2y2z, \
    template_fuse_reduction


# Note: this is a short-term solution. In the long run we need to think about how to reuse pytest
def benchmark_async(func):
    def body():
        for arch in [ti.cpu, ti.cuda]:
            for async_mode in [True, False]:
                os.environ['TI_CURRENT_BENCHMARK'] = func.__name__
                ti.init(arch=arch, async_mode=async_mode)
                func()
    return body

@benchmark_async
def fuse_dense_x2y2z():
    template_fuse_dense_x2y2z(size=100 * 1024**2,
                              repeat=10,
                              benchmark_repeat=10,
                              benchmark=True)


@benchmark_async
def fuse_reduction():
    template_fuse_reduction(size=100 * 1024**2,
                            repeat=10,
                            benchmark_repeat=10,
                            benchmark=True)
    
@benchmark_async
def fill_1d():
    a = ti.field(dtype=ti.f32, shape=100 * 1024**2)
    
    @ti.kernel
    def fill():
        for i in a:
            a[i] = 1.0
    
    return ti.benchmark(fill, repeat=100)

@benchmark_async
def fill_scalar():
    a = ti.field(dtype=ti.f32, shape=())
    
    @ti.kernel
    def fill():
        a[None] = 1.0
    
    return ti.benchmark(fill, repeat=1000)
