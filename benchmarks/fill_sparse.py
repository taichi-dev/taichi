import taichi as ti


def benchmark_nested_struct():
    a = ti.var(dt=ti.f32)
    N = 512

    @ti.layout
    def place():
        ti.root.pointer(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in ti.ndrange(N * 8, N * 8):
            a[i, j] = 2.0

    fill()

    return ti.benchmark(fill)


def benchmark_nested_struct_fill_and_clear():
    a = ti.var(dt=ti.f32)
    N = 512

    @ti.layout
    def place():
        ti.root.pointer(ti.ij, [N, N]).dense(ti.ij, [8, 8]).place(a)

    @ti.kernel
    def fill():
        for i, j in ti.ndrange(N * 8, N * 8):
            a[i, j] = 2.0

    @ti.kernel
    def clear():
        for i, j in a.parent():
            ti.deactivate(a.parent().parent(), [i, j])

    def task():
        fill()
        clear()

    return ti.benchmark(task, repeat=30)


'''
ti.init(arch=ti.cuda, enable_profiler=True)
benchmark_nested_struct_fill_and_clear()
ti.profiler_print()
'''
