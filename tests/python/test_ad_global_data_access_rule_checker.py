import pytest
from numpy import float16

import taichi as ti
from tests import test_utils


@test_utils.test(check_autodiff_valid=True, debug=True)
def test_adjoint_flag_needs_grad():
    x = ti.field(float, shape=(), needs_grad=True)

    @ti.kernel
    def test():
        x[None] = 1

    test()

    assert x.snode.ptr.has_adjoint_flag()


@test_utils.test(check_autodiff_valid=True, debug=True)
def test_adjoint_flag_lazy_grad():
    x = ti.field(float, shape=())
    ti.root.lazy_grad()

    @ti.kernel
    def test():
        x[None] = 1

    test()

    assert x.snode.ptr.has_adjoint_flag()


@test_utils.test(check_autodiff_valid=True, debug=True)
def test_adjoint_flag_place_grad():
    x = ti.field(float)
    y = ti.field(float)
    ti.root.place(x, x.grad, y)

    @ti.kernel
    def test():
        x[None] = 1

    test()

    assert x.snode.ptr.has_adjoint_flag()
    assert not y.snode.ptr.has_adjoint_flag()


@test_utils.test(require=ti.extension.assertion,
                 exclude=[ti.cc],
                 check_autodiff_valid=True,
                 debug=True)
def test_break_gdar_rule_1():
    N = 16
    x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    b = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def func_broke_rule_1():
        loss[None] = x[1] * b[None]
        b[None] += 100

    for i in range(N):
        x[i] = i

    b[None] = 10
    loss.grad[None] = 1

    with pytest.raises(ti.TaichiAssertionError):
        with ti.Tape(loss):
            func_broke_rule_1()
