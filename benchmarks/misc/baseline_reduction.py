from utils import dtype_size, scale_repeat

import taichi as ti


def reduction(arch, dtype, dsize, repeat=10):

    repeat = scale_repeat(arch, dsize, repeat)
    n = dsize // dtype_size[dtype]

    ## fill x
    x = ti.field(dtype, shape=n)

    if dtype in [ti.f32, ti.f64]:

        @ti.kernel
        def fill_const(n: ti.i32):
            for i in range(n):
                x[i] = 0.1
    else:

        @ti.kernel
        def fill_const(n: ti.i32):
            for i in range(n):
                x[i] = 1

    # compile the kernel first
    fill_const(n)

    ## reduce
    y = ti.field(dtype, shape=())
    if dtype in [ti.f32, ti.f64]:
        y[None] = 0.0
    else:
        y[None] = 0

    @ti.kernel
    def reduction(n: ti.i32):
        for i in range(n):
            y[None] += ti.atomic_add(y[None], x[i])

    # compile the kernel first
    reduction(n)
    ti.sync()
    ti.kernel_profiler_clear()
    ti.sync()
    for i in range(repeat):
        reduction(n)
    ti.sync()
    kernelname = reduction.__name__
    suffix = "_c"
    quering_result = ti.query_kernel_profiler(kernelname + suffix)
    return quering_result.min
