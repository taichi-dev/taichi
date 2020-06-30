import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=True)
x, y = ti.var(ti.f32), ti.var(ti.f32)

N = 4096
bs = 16

block = ti.root.pointer(ti.ij, N // bs)
block.dense(ti.ij, bs).place(x)
block.dense(ti.ij, bs).place(y)


@ti.kernel
def populate():
    for i in range(bs, N - bs):
        for j in range(bs, N - bs):
            x[i, j] = 1

@ti.kernel
def laplace():
    ti.cache_shared(x)
    for i, j in x:
            y[i, j] = 4.0 * x[i, j] - x[i - 1,
                                        j] - x[i + 1,
                                               j] - x[i, j - 1] - x[i, j + 1]



populate()
for i in range(10):
    laplace()

ti.kernel_profiler_print()
