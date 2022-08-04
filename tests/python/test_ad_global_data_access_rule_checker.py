import pytest
from numpy import float16

import taichi as ti
from tests import test_utils


@test_utils.test(debug=True)
def test_adjoint_visited_needs_grad():
    x = ti.field(float, shape=(), needs_grad=True)

    @ti.kernel
    def test():
        x[None] = 1

    test()

    assert x.snode.ptr.has_adjoint_visited()


@test_utils.test(debug=True)
def test_adjoint_visited_lazy_grad():
    x = ti.field(float, shape=())
    ti.root.lazy_grad()

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

    @ti.kernel
    def test():
        x[None] = 1

    test()

    assert x.snode.ptr.has_adjoint_visited()
    assert not y.snode.ptr.has_adjoint_visited()


@test_utils.test(debug=False)
def test_adjoint_visited_needs_grad():
    x = ti.field(float, shape=(), needs_grad=True)

    @ti.kernel
    def test():
        x[None] = 1

    with pytest.warns(Warning) as record:
        with ti.ad.Tape(loss=x, validation=True):
            test()

    warn_raised = False
    for warn in record:
        if 'Debug mode is disabled, autodiff valid check will not work. Please specify `ti.init(debug=True)` to enable the check.' in warn.message.args[
                0]:
            warn_raised = True
    assert warn_raised
