import taichi as ti
from taichi import approx


def test_violate_kernel_simplicity1():
    ti.init()
    x = ti.var(ti.f32)
    y = ti.var(ti.f32)

    ti.root.dense(ti.i, 1).place(x, y)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in x:
            y[i] = x[i] * 42
            for j in ti.static(range(3)):
                y[i] += x[i]

    func()
    func.grad()


def test_violate_kernel_simplicity2():
    ti.init()
    x = ti.var(ti.f32)
    y = ti.var(ti.f32)

    ti.root.dense(ti.i, 1).place(x, y)
    ti.root.lazy_grad()

    @ti.kernel
    def func():
        for i in x:
            for j in ti.static(range(3)):
                y[i] += x[i]
            y[i] += x[i] * 42

    func()
    func.grad()

#  ti test ad_basics -a x64 -t 1 -v