import taichi as ti

ti.init()

x = ti.var(ti.i32)
y = ti.var(ti.i32)

ti.root.pointer(ti.ij, 4).dense(ti.ij, 8).place(x, y)


@ti.kernel
def copy():
    for i, j in y:
        x[i, j] = y[i, j]


copy()
