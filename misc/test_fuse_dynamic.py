import taichi as ti

ti.init()

x = ti.var(ti.i32)
y = ti.var(ti.i32)
z = ti.var(ti.i32)

ti.root.dynamic(ti.i, 1048576, chunk_size=2048).place(x, y, z)


@ti.kernel
def x_to_y():
    for i in x:
        y[i] = x[i] + 1


@ti.kernel
def y_to_z():
    for i in x:
        z[i] = y[i] + 1


n = 10000

for i in range(n):
    x[i] = i * 10

x_to_y()
y_to_z()

for i in range(n):
    x[i] = i * 10
    assert y[i] == x[i] + 1
    assert z[i] == x[i] + 2
