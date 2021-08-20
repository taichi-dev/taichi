import taichi as ti

N = 1024**3 // 4  # 1 GB per buffer


# 4 B/it
@ti.test(exclude=ti.opengl)
def benchmark_memset():
    a = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def memset():
        for i in a:
            a[i] = 1.0

    return ti.benchmark(memset, repeat=10)


# 8 B/it
@ti.test(exclude=ti.opengl)
def benchmark_sscal():
    a = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def task():
        for i in a:
            a[i] = 0.5 * a[i]

    return ti.benchmark(task, repeat=10)


# 8 B/it
@ti.test(exclude=ti.opengl)
def benchmark_memcpy():
    a = ti.field(dtype=ti.f32, shape=N)
    b = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def memcpy():
        for i in a:
            a[i] = b[i]

    return ti.benchmark(memcpy, repeat=10)


# 12 B/it
@ti.test(exclude=ti.opengl)
def benchmark_saxpy():
    x = ti.field(dtype=ti.f32, shape=N)
    y = ti.field(dtype=ti.f32, shape=N)
    z = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def task():
        for i in x:
            a = 123
            z[i] = a * x[i] + y[i]

    return ti.benchmark(task, repeat=10)
