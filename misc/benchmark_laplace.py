import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=True)

dim = 3

x, y, y2 = ti.var(ti.i32), ti.var(ti.i32), ti.var(ti.i32)


if dim == 2:
    index = ti.ij
    N = 1024
    bs = 16
elif dim == 3:
    index = ti.ijk
    N = 256
    bs = 8
else:
    assert False
    
block = ti.root.pointer(index, N // bs)

block.dense(index, bs).place(x)
block.dense(index, bs).place(y)
block.dense(index, bs).place(y2)

ndrange = ((bs, N - bs),) * dim
stencil_range = ((-1, 2),) * dim

@ti.kernel
def populate():
    for I in ti.grouped(ti.ndrange(*ndrange)):
        x[I] = I.sum()

@ti.kernel
def laplace(use_bls: ti.template(), y: ti.template()):
    if ti.static(use_bls):
        ti.cache_shared(x)
    ti.block_dim(bs ** dim)
    for I in ti.grouped(x):
        s = 0
        for offset in ti.static(ti.grouped(ti.ndrange(*stencil_range))):
            s = s + x[I + offset]
        y[I] = s



populate()
for i in range(10):
    laplace(False, y2)
for i in range(10):
    laplace(True, y)


ti.kernel_profiler_print()

for i in range(N):
    for j in range(N):
        if y[i, j] != y2[i, j]:
            print(i, j, y[i, j], y2[i, j])
        assert y[i, j] == y2[i, j]
