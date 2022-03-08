import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_function_without_return():
    x = ti.field(ti.i32, shape=())

    @ti.experimental.real_func
    def foo(val: ti.i32):
        x[None] += val

    @ti.kernel
    def run():
        foo(40)
        foo(2)

    x[None] = 0
    run()
    assert x[None] == 42


# @test_utils.test(arch=[ti.cpu, ti.gpu])
# def test_function_with_return():
#     x = ti.field(ti.i32, shape=())
#
#     @ti.experimental.real_func
#     def foo(val: ti.i32) -> ti.i32:
#         x[None] += val
#         return val
#
#     @ti.kernel
#     def run():
#         a = foo(40)
#         foo(2)
#         assert a == 40
#
#     x[None] = 0
#     run()
#     assert x[None] == 42
#
#
# @test_utils.test(arch=[ti.cpu, ti.gpu])
# def test_call_expressions():
#     x = ti.field(ti.i32, shape=())
#
#     @ti.experimental.real_func
#     def foo(val: ti.i32) -> ti.i32:
#         if x[None] > 10:
#             x[None] += 1
#         x[None] += val
#         return 0
#
#     @ti.kernel
#     def run():
#         assert foo(15) == 0
#         assert foo(10) == 0
#
#     x[None] = 0
#     run()
#     assert x[None] == 26
#
#


@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_default_templates():
    @ti.func
    def func1(x: ti.template()):
        x = 1

    @ti.func
    def func2(x: ti.template()):
        x += 1

    @ti.func
    def func3(x):
        x = 1

    @ti.func
    def func4(x):
        x += 1

    @ti.func
    def func1_field(x: ti.template()):
        x[None] = 1

    @ti.func
    def func2_field(x: ti.template()):
        x[None] += 1

    @ti.func
    def func3_field(x):
        x[None] = 1

    @ti.func
    def func4_field(x):
        x[None] += 1

    v = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def run_func():
        a = 0
        func1(a)
        assert a == 1
        b = 0
        func2(b)
        assert b == 1
        c = 0
        func3(c)
        assert c == 0
        d = 0
        func4(d)
        assert d == 0

        v[None] = 0
        func1_field(v)
        assert v[None] == 1
        v[None] = 0
        func2_field(v)
        assert v[None] == 1
        v[None] = 0
        func3_field(v)
        assert v[None] == 1
        v[None] = 0
        func4_field(v)
        assert v[None] == 1

    run_func()


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_experimental_templates():
    x = ti.field(ti.i32, shape=())
    y = ti.field(ti.i32, shape=())
    answer = ti.field(ti.i32, shape=8)

    @ti.kernel
    def kernel_inc(x: ti.template()):
        x[None] += 1

    def run_kernel():
        x[None] = 10
        y[None] = 20
        kernel_inc(x)
        assert x[None] == 11
        assert y[None] == 20
        kernel_inc(y)
        assert x[None] == 11
        assert y[None] == 21

    @ti.experimental.real_func
    def inc(x: ti.template()):
        x[None] += 1

    @ti.kernel
    def run_func():
        x[None] = 10
        y[None] = 20
        inc(x)
        answer[0] = x[None]
        answer[1] = y[None]
        inc(y)
        answer[2] = x[None]
        answer[3] = y[None]

    def verify():
        assert answer[0] == 11
        assert answer[1] == 20
        assert answer[2] == 11
        assert answer[3] == 21

    run_kernel()
    run_func()
    verify()


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_missing_arg_annotation():
    with pytest.raises(ti.TaichiSyntaxError, match='must be type annotated'):

        @ti.experimental.real_func
        def add(a, b: ti.i32) -> ti.i32:
            return a + b


@test_utils.test(arch=[ti.cpu, ti.gpu])
def test_missing_return_annotation():
    with pytest.raises(ti.TaichiCompilationError,
                       match='return value must be annotated'):

        @ti.experimental.real_func
        def add(a: ti.i32, b: ti.i32):
            return a + b

        @ti.kernel
        def run():
            add(30, 2)

        run()
