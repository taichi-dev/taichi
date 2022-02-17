"""
import taichi as ti

<<<<<<< HEAD
ti.cfg.print_ir = True
=======
ti.lang.impl.current_cfg().print_ir = True
>>>>>>> 5d372d76cdb12826fd31d3f6bd81b56ed22bcef7


def test_global_store_branching():
    # ti.reset()

    N = 16
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    ti.root.dense(ti.i, N).place(x)
    ti.root.dense(ti.i, N).place(y)
    ti.root.lazy_grad()

    @ti.kernel
    def oldeven():
        for i in range(N):
            if i % 2 == 0:
                x[i] = y[i]

    for i in range(N):
        x.grad[i] = 1

    oldeven()
    oldeven.grad()

    for i in range(N):
        assert y.grad[i] == (i % 2 == 0)
"""
