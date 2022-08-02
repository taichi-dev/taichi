from numpy import float16

import taichi as ti
from tests import test_utils


@test_utils.test(debug=True)
def test_adjoint_visited_needs_grad():
    x = ti.field(float, shape=(), needs_grad=True)
    ti.root.root._allocate_grad_visited()

    @ti.kernel
    def test():
        x[None] = 1

    test()

    assert x.snode.ptr.has_adjoint_visited()


@test_utils.test(debug=True)
def test_adjoint_visited_lazy_grad():
    x = ti.field(float, shape=())
    ti.root.lazy_grad()
    ti.root.root._allocate_grad_visited()

    @ti.kernel
    def test():
        x[None] = 1

    test()

    assert x.snode.ptr.has_adjoint_visited()


@test_utils.test(debug=True)
def test_adjoint_visited_place_grad():
    x = ti.field(float)
    y = ti.field(float)
    ti.root.place(x, x.grad, y)
    ti.root.root._allocate_grad_visited()

    @ti.kernel
    def test():
        x[None] = 1

    test()

    assert x.snode.ptr.has_adjoint_visited()
    assert not y.snode.ptr.has_adjoint_visited()
