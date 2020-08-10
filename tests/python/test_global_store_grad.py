"""
import taichi as ti

ti.cfg.print_ir = True


def test_global_store_branching():
    # ti.reset()

    N = 16
    ti.runtime.print_preprocessed = True
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
