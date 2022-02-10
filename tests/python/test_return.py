import pytest

import taichi as ti
from tests import test_utils
from tests.test_utils import approx


@test_utils.test()
def test_return_without_type_hint():
    @ti.kernel
    def kernel():
        return 1

    with pytest.raises(ti.TaichiCompilationError):
        kernel()


def test_const_func_ret():
    ti.init()

    @ti.kernel
    def func1() -> ti.f32:
        return 3

    @ti.kernel
    def func2() -> ti.i32:
        return 3.3  # return type mismatch, will be auto-casted into ti.i32

    assert func1() == approx(3)
    assert func2() == 3


@test_utils.test()
def _test_binary_func_ret(dt1, dt2, dt3, castor):
    @ti.kernel
    def func(a: dt1, b: dt2) -> dt3:
        return a * b

    if ti.types.is_integral(dt1):
        xs = list(range(4))
    else:
        xs = [0.2, 0.4, 0.8, 1.0]

    if ti.types.is_integral(dt2):
        ys = list(range(4))
    else:
        ys = [0.2, 0.4, 0.8, 1.0]

    for x, y in zip(xs, ys):
        assert func(x, y) == approx(castor(x * y))


def test_binary_func_ret():
    _test_binary_func_ret(ti.i32, ti.f32, ti.f32, float)
    _test_binary_func_ret(ti.f32, ti.i32, ti.f32, float)
    _test_binary_func_ret(ti.i32, ti.f32, ti.i32, int)
    _test_binary_func_ret(ti.f32, ti.i32, ti.i32, int)


@test_utils.test()
def test_return_in_static_if():
    @ti.kernel
    def foo(a: ti.template()) -> ti.i32:
        if ti.static(a == 1):
            return 1
        elif ti.static(a == 2):
            return 2
        return 3

    assert foo(1) == 1
    assert foo(2) == 2
    assert foo(123) == 3


@test_utils.test()
def test_func_multiple_return():
    @ti.func
    def safe_sqrt(a):
        if a > 0:
            return ti.sqrt(a)
        else:
            return 0.0

    @ti.kernel
    def kern(a: float):
        print(safe_sqrt(a))

    with pytest.raises(
            ti.TaichiCompilationError,
            match='Return inside non-static if/for is not supported'):
        kern(-233)


@test_utils.test()
def test_return_inside_static_for():
    @ti.kernel
    def foo() -> ti.i32:
        a = 0
        for i in ti.static(range(10)):
            a += i * i
            if ti.static(i == 8):
                return a

    assert foo() == 204


@test_utils.test()
def test_return_inside_non_static_for():
    with pytest.raises(
            ti.TaichiCompilationError,
            match='Return inside non-static if/for is not supported'):

        @ti.kernel
        def foo() -> ti.i32:
            for i in range(10):
                return i

        foo()


@test_utils.test()
def test_kernel_no_return():
    with pytest.raises(
            ti.TaichiSyntaxError,
            match=
            "Kernel has a return type but does not have a return statement"):

        @ti.kernel
        def foo() -> ti.i32:
            pass

        foo()


@test_utils.test()
def test_func_no_return():
    with pytest.raises(
            ti.TaichiCompilationError,
            match=
            "Function has a return type but does not have a return statement"):

        @ti.func
        def bar() -> ti.i32:
            pass

        @ti.kernel
        def foo() -> ti.i32:
            return bar()

        foo()
