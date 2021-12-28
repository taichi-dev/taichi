from utils import dtype_size, scaled_repeat_times

import taichi as ti


def init_const(x, dtype, num_elements):

    @ti.kernel
    def init_const(x: ti.template(), n: ti.i32):
        for i in range(n):
            x[i] = ti.cast(0.7, dtype)

    init_const(x, num_elements)


def membound_benchmark(func, num_elements, repeat):
    # compile the kernel first
    func(num_elements)
    ti.clear_kernel_profile_info()
    for i in range(repeat):
        func(num_elements)
    kernelname = func.__name__
    quering_result = ti.query_kernel_profile_info(kernelname)
    return quering_result.min


def fill(arch, dtype, dsize, repeat=10):

    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype)

    x = ti.field(dtype, shape=num_elements)

    @ti.kernel
    def fill_const(n: ti.i32):
        for i in range(n):
            x[i] = ti.cast(0.7, dtype)

    return membound_benchmark(fill_const, num_elements, repeat)


def saxpy(arch, dtype, dsize, repeat=10):

    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype) // 3  #z=x+y

    x = ti.field(dtype, shape=num_elements)
    y = ti.field(dtype, shape=num_elements)
    z = ti.field(dtype, shape=num_elements)

    @ti.kernel
    def saxpy(n: ti.i32):
        for i in range(n):
            z[i] = 17 * x[i] + y[i]

    init_const(x, dtype, num_elements)
    init_const(y, dtype, num_elements)

    return membound_benchmark(saxpy, num_elements, repeat)


def reduction(arch, dtype, dsize, repeat=10):

    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype)

    x = ti.field(dtype, shape=num_elements)
    y = ti.field(dtype, shape=())
    y[None] = 0

    @ti.kernel
    def reduction(n: ti.i32):
        for i in range(n):
            y[None] += x[i]

    init_const(x, dtype, num_elements)
    return membound_benchmark(reduction, num_elements, repeat)


memory_bound_cases_list = [fill, saxpy, reduction]
