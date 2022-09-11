import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(debug=True, validate_autodiff=True, exclude=[ti.cc])
def test_adjoint_checkbit_needs_grad():
    x = ti.field(float, shape=(), needs_grad=True)

    @ti.kernel
    def test():
        x[None] = 1

    with ti.ad.Tape(loss=x, validation=True):
        test()

    assert x.snode.ptr.has_adjoint_checkbit()


@test_utils.test(debug=True, validate_autodiff=True, exclude=[ti.cc])
def test_adjoint_checkbit_lazy_grad():
    x = ti.field(float, shape=())
    ti.root.lazy_grad()

    @ti.kernel
    def test():
        x[None] = 1

    with ti.ad.Tape(loss=x, validation=True):
        test()

    assert x.snode.ptr.has_adjoint_checkbit()


@test_utils.test(debug=True, validate_autodiff=True, exclude=[ti.cc])
def test_adjoint_checkbit_place_grad():
    x = ti.field(float)
    y = ti.field(float)
    ti.root.place(x, x.grad, y)

    @ti.kernel
    def test():
        x[None] = 1

    with ti.ad.Tape(loss=x, validation=True):
        test()

    assert x.snode.ptr.has_adjoint_checkbit()
    assert not y.snode.ptr.has_adjoint_checkbit()


@test_utils.test(debug=False, validate_autodiff=True)
def test_adjoint_checkbit_needs_grad():
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


@test_utils.test(require=ti.extension.assertion,
                 exclude=[ti.cc],
                 debug=True,
                 validate_autodiff=True)
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
        with ti.ad.Tape(loss=loss, validation=True):
            func_broke_rule_1()
