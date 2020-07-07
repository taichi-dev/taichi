import taichi as ti

ti.init(arch=ti.cuda, print_ir=True)

x, y, z, z2 = ti.var(ti.i32), ti.var(ti.i32), ti.var(ti.i32), ti.var(ti.i32)

N = 128
bs = 8

ti.root.pointer(ti.ij, N // bs).dense(ti.ij, bs).place(x, y, z, z2)

@ti.kernel
def populate():
    for i, j in ti.ndrange((bs, N - bs), (bs, N - bs)):
        x[i, j] = i - j
        y[i, j] = i + j * j

@ti.kernel
def copy(bls: ti.template(), z: ti.template()):
    if ti.static(bls):
        ti.cache_shared(x, y)
    for i, j in x:
        z[i, j] = x[i, j - 2] + y[i + 2, j - 1] + y[i - 1, j]

populate()
copy(False, z2)
copy(True, z)

for i in range(N):
    for j in range(N):
        if z[i, j] != z2[i, j]:
            print(i, j, z[i, j], z2[i, j])
        assert z[i, j] == z2[i, j]
