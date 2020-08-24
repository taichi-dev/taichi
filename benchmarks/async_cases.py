import taichi as ti
import os
import sys
import functools

sys.path.append(os.path.join(ti.core.get_repo_dir(), 'tests', 'python'))

from fuse_test_template import template_fuse_dense_x2y2z, \
    template_fuse_reduction


# Note: this is a short-term solution. In the long run we need to think about how to reuse pytest
def benchmark_async(func):
    @functools.wraps(func)
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

    ti.benchmark(fill, repeat=100)


@benchmark_async
def fill_scalar():
    a = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def fill():
        a[None] = 1.0

    ti.benchmark(fill, repeat=1000)


@benchmark_async
def sparse_numpy():
    a = ti.field(dtype=ti.f32)
    b = ti.field(dtype=ti.f32)

    block_count = 64
    block_size = 32
    # a, b always share the same sparsity
    ti.root.pointer(ti.ij, block_count).dense(ti.ij, block_size).place(a, b)

    @ti.kernel
    def initialize():
        for i, j in ti.ndrange(block_count * block_size,
                               block_count * block_size):
            if (i // block_size + j // block_size) % 4 == 0:
                a[i, j] = i + j

    @ti.kernel
    def saxpy(x: ti.template(), y: ti.template(), alpha: ti.f32):
        for i, j in x:
            y[i, j] = alpha * x[i, j] + y[i, j]

    def task():
        saxpy(a, b, 2)
        saxpy(b, a, 1.1)
        saxpy(b, a, 1.1)
        saxpy(a, b, 1.1)
        saxpy(a, b, 1.1)
        saxpy(a, b, 1.1)

    ti.benchmark(task, repeat=100)


with_autodiff = False  # For some reason autodiff crashes with async.
if with_autodiff:

    @benchmark_async
    def autodiff():

        n = 1024**2 * 10

        a = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
        b = ti.field(dtype=ti.f32, shape=n)
        loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        @ti.kernel
        def compute_loss():
            for i in a:
                loss[None] += a[i]

        @ti.kernel
        def accumulate_grad():
            for i in a:
                b[i] += a.grad[i]

        def task():
            with ti.Tape(loss=loss):
                # The forward kernel of compute_loss should be completely eliminated (except for the last one)
                compute_loss()

            accumulate_grad()

        ti.benchmark(task, repeat=100)


@benchmark_async
def stencil_reduction():
    a = ti.field(dtype=ti.f32)
    b = ti.field(dtype=ti.f32)
    total = ti.field(dtype=ti.f32, shape=())

    block_count = 1024
    block_size = 1024
    # a, b always share the same sparsity
    ti.root.pointer(ti.i, block_count).dense(ti.i, block_size).place(a, b)

    @ti.kernel
    def initialize():
        for i in range(block_size, (block_size - 1) * block_count):
            a[i] = i

    @ti.kernel
    def stencil():
        for i in a:
            b[i] = a[i - 1] + a[i] + a[i + 1]

    @ti.kernel
    def reduce():
        for i in a:
            total[None] += b[i]

    def task():
        for i in range(2):
            initialize()
            stencil()
            reduce()

    ti.benchmark(task, repeat=100)


# TODO: add mpm_breakdown
