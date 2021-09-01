import numpy as np
import pytest

import taichi as ti


@ti.test()
def test_no_grad():
    x = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    N = 1

    # no gradients allocated for x
    ti.root.dense(ti.i, N).place(x)
    ti.root.place(loss, loss.grad)

    @ti.kernel
    def func():
        for i in range(N):
            ti.atomic_add(loss[None], x[i]**2)

    with ti.Tape(loss):
        func()


@ti.test()
def test_raise_no_gradient():
    y = ti.field(shape=(), name='y', dtype=ti.f32, needs_grad=True)
    x = ti.field(shape=(), name='x', dtype=ti.f32)
    z = np.array([1.0])

    @ti.kernel
    def func(x: ti.template()):
        y[None] = x.grad[None] * x.grad[None]
        z[0] = x.grad[None]

    x[None] = 5.
    with pytest.raises(RuntimeError) as e:
        func(x)

    assert e.type is RuntimeError
    assert e.value.args[
        0] == f"Gradient x.grad has not been placed, check whether `needs_grad=True`"
