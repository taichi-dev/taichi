import taichi as ti

# originally by @KLozes


@ti.test()
def benchmark_flat_struct():
    N = 4096
    a = ti.field(dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def fill():
        for i, j in a:
            a[i, j] = 2.0

    return ti.benchmark(fill, repeat=500)


@ti.test()
def benchmark_flat_range():
    N = 4096
    a = ti.field(dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def fill():
        for i, j in ti.ndrange(N, N):
            a[i, j] = 2.0

    return ti.benchmark(fill, repeat=700)


@ti.test()
def benchmark_nested_struct():
    a = ti.field(dtype=ti.f32)
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in a:
            a[i, j] = 2.0

    return ti.benchmark(fill, repeat=700)


@ti.test()
def benchmark_nested_struct_listgen_8x8():
    a = ti.field(dtype=ti.f32)
    ti.cfg.demote_dense_struct_fors = False
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in a:
            a[i, j] = 2.0

    return ti.benchmark(fill, repeat=1000)


@ti.test()
def benchmark_nested_struct_listgen_16x16():
    a = ti.field(dtype=ti.f32)
    ti.cfg.demote_dense_struct_fors = False
    N = 256

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [16, 16]).place(a)

    @ti.kernel
    def fill():
        for i, j in a:
            a[i, j] = 2.0

    return ti.benchmark(fill, repeat=700)


@ti.test()
def benchmark_nested_range_blocked():
    a = ti.field(dtype=ti.f32)
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for X in range(N * N):
            for Y in range(64):
                a[X // N * 8 + Y // 8, X % N * 8 + Y % 8] = 2.0

    return ti.benchmark(fill, repeat=800)


@ti.test()
def benchmark_nested_range():
    a = ti.field(dtype=ti.f32)
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for j in range(N * 8):
            for i in range(N * 8):
                a[i, j] = 2.0

    return ti.benchmark(fill, repeat=1000)


@ti.test()
def benchmark_root_listgen():
    a = ti.field(dtype=ti.f32)
    ti.cfg.demote_dense_struct_fors = False
    N = 512

    ti.root.dense(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in a.parent():
            a[i, j] = 2.0

    return ti.benchmark(fill, repeat=800)
