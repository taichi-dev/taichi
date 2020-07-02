import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=True)#, print_ir=True)
x, y, y2 = ti.var(ti.i32), ti.var(ti.i32), ti.var(ti.i32)

N = 1024
bs = 8

block = ti.root.pointer(ti.ij, N // bs)
block.dense(ti.ij, bs).place(x)
block.dense(ti.ij, bs).place(y)
block.dense(ti.ij, bs).place(y2)


@ti.kernel
def populate():
    for i in range(bs, N - bs):
        for j in range(bs, N - bs):
            x[i, j] = j

@ti.kernel
def laplace(use_bls: ti.template(), y: ti.template()):
    if ti.static(use_bls):
        ti.cache_shared(x)
    for i, j in x:
        # y[i, j] = x[i + 3, j - 1]#  - x[i, j]
        # print(x[i, j + 1], x[i, j])
        # y[i, j] = x[i, j + 1] - x[i, j]
        # y[i, j] = - x[i, j]
        y[i, j] = 4.0 * x[i, j] - x[i - 1,
                                    j] - x[i + 1,
                                           j] - x[i, j - 1] - x[i, j + 1]



populate()
for i in range(10):
    laplace(False, y2)
for i in range(10):
    laplace(True, y)
    
for i in range(N):
    for j in range(N):
        if y[i, j] != y2[i, j]:
            print(i, j, y[i, j], y2[i, j])
        assert y[i, j] == y2[i, j]

ti.kernel_profiler_print()
