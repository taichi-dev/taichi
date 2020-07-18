import taichi as ti

ti.init()

x = ti.var(ti.i32)
y = ti.var(ti.i32)
n = 128

ti.root.dynamic(ti.i, n, 32).place(x)
ti.root.dynamic(ti.i, n, 32).place(y)

@ti.kernel
def func():
    for i in range(n):
        u = ti.append(x.parent(), [], i)
        y[u] = i + 1

func()

for i in range(n):
    assert x[i] + 1 == y[i]
