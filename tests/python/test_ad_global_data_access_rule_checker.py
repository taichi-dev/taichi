import pytest
from taichi.lang.enums import AutodiffMode
from taichi.lang.kernel_impl import _kernel_impl

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


@test_utils.test(require=ti.extension.assertion,
                 exclude=[ti.cc],
                 debug=True,
                 validate_autodiff=True)
def test_skip_grad_replaced():
    N = 16
    x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    b = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    # This kernel breaks the global data access rule
    @ti.kernel
    def kernel_1():
        loss[None] = x[1] * b[None]
        b[None] += 100

    @ti.ad.grad_replaced
    def kernel_2():
        loss[None] = x[1] * b[None]
        b[None] += 100

    # The user defined grad kernel is not restricted by the global data access rule, thus should be skipped when checking
    @ti.ad.grad_for(kernel_2)
    def kernel_2_grad():
        pass

    for i in range(N):
        x[i] = i

    b[None] = 10
    loss.grad[None] = 1

    with pytest.raises(ti.TaichiAssertionError):
        with ti.ad.Tape(loss=loss, validation=True):
            kernel_1()

    with ti.ad.Tape(loss=loss, validation=True):
        kernel_2()


@test_utils.test(require=ti.extension.assertion,
                 exclude=[ti.cc],
                 debug=True,
                 validate_autodiff=True)
def test_autodiff_mode_recovered():
    N = 16
    x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    b = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def kernel_1():
        loss[None] = x[1] * b[None]

    @ti.kernel
    def kernel_2():
        loss[None] = x[1] * b[None]

    for i in range(N):
        x[i] = i

    b[None] = 10
    loss.grad[None] = 1

    func_calls = []
    with ti.ad.Tape(loss=loss, validation=True) as t:
        kernel_1()
        kernel_2()
        for f, _ in t.calls:
            assert f.autodiff_mode == AutodiffMode.VALIDATION
        func_calls = t.calls
    for f, _ in func_calls:
        assert f.autodiff_mode == AutodiffMode.NONE

    # Test for kernels whose initial modes are AutodiffMode.REVERSE
    def func_3():
        loss[None] = x[1] * b[None]

    def func_4():
        loss[None] = x[1] * b[None]

    kernel_3 = _kernel_impl(func_3, level_of_class_stackframe=3)
    kernel_3._primal.autodiff_mode = AutodiffMode.REVERSE

    kernel_4 = _kernel_impl(func_4, level_of_class_stackframe=3)
    kernel_4._primal.autodiff_mode = AutodiffMode.REVERSE

    func_calls = []
    with ti.ad.Tape(loss=loss, validation=True) as t:
        kernel_3()
        kernel_4()
        for f, _ in t.calls:
            assert f.autodiff_mode == AutodiffMode.VALIDATION
        func_calls = t.calls
    for f, _ in func_calls:
        assert f.autodiff_mode == AutodiffMode.REVERSE
