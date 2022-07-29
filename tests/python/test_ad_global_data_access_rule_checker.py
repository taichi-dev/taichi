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
