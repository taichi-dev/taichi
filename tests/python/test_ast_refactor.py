import pytest

import taichi as ti
from taichi import approx


@ti.test(experimental_ast_refactor=True)
def test_binop():
    @ti.kernel
    def foo(x: ti.i32, y: ti.i32, a: ti.template()):
        a[0] = x + y
        a[1] = x - y
        a[2] = x * y
        a[3] = ti.ti_float(x) / y
        a[4] = x // y
        a[5] = x % y
        a[6] = x**y
        a[7] = x << y
        a[8] = x >> y
        a[9] = x | y
        a[10] = x ^ y
        a[11] = x & y

    x = 37
    y = 5
    a = ti.field(ti.f32, shape=(12, ))
    b = ti.field(ti.f32, shape=(12, ))

    a[0] = x + y
    a[1] = x - y
    a[2] = x * y
    a[3] = x / y
    a[4] = x // y
    a[5] = x % y
    a[6] = x**y
    a[7] = x << y
    a[8] = x >> y
    a[9] = x | y
    a[10] = x ^ y
    a[11] = x & y

    foo(x, y, b)

    for i in range(12):
        assert a[i] == approx(b[i])


@ti.test(experimental_ast_refactor=True)
def test_unaryop():
    @ti.kernel
    def foo(x: ti.i32, a: ti.template()):
        a[0] = +x
        a[1] = -x
        a[2] = not x
        a[3] = ~x

    x = 1234
    a = ti.field(ti.i32, shape=(4, ))
    b = ti.field(ti.i32, shape=(4, ))

    a[0] = +x
    a[1] = -x
    a[2] = not x
    a[3] = ~x

    foo(x, b)

    for i in range(4):
        assert a[i] == b[i]


@ti.test(experimental_ast_refactor=True)
def test_return():
    @ti.kernel
    def foo(x: ti.i32) -> ti.i32:
        return x + 1

    assert foo(1) == 2


@ti.test(experimental_ast_refactor=True, print_preprocessed_ir=True)
def test_if():
    @ti.kernel
    def foo(x: ti.i32) -> ti.i32:
        ret = 0
        if x:
            ret = 1
        else:
            ret = 0
        return ret

    assert foo(1)
    assert not foo(0)


@ti.test(experimental_ast_refactor=True, print_preprocessed_ir=True)
def test_static_if():
    @ti.kernel
    def foo(x: ti.template()) -> ti.i32:
        ret = 0
        if ti.static(x):
            ret = 1
        else:
            ret = 0
        return ret

    assert foo(1)
    assert not foo(0)


@ti.test(experimental_ast_refactor=True, print_preprocessed_ir=True)
def test_struct_for():
    a = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo(x: ti.i32):
        for i in a:
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        assert a[i] == 5


@ti.test(experimental_ast_refactor=True, print_preprocessed_ir=True)
def test_static_for():
    a = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo(x: ti.i32):
        for i in ti.static(range(10)):
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        assert a[i] == 5


@ti.test(experimental_ast_refactor=True, print_preprocessed_ir=True)
def test_func():
    @ti.func
    def bar(x):
        return x * x, -x

    a = ti.field(ti.i32, shape=(10, ))
    b = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo():
        for i in a:
            a[i], b[i] = bar(i)

    foo()
    for i in range(10):
        assert a[i] == i * i
        assert b[i] == -i


@ti.test(experimental_ast_refactor=True, print_preprocessed_ir=True)
def test_func_in_python_func():
    @ti.func
    def bar(x: ti.template()):
        if ti.static(x):
            mat = bar(x // 2)
            mat = mat @ mat
            if ti.static(x % 2):
                mat = mat @ ti.Matrix([[1, 1], [1, 0]])
            return mat
        else:
            return ti.Matrix([[1, 0], [0, 1]])

    def fibonacci(x):
        return ti.subscript(bar(x), 1, 0)

    @ti.kernel
    def foo(x: ti.template()) -> ti.i32:
        return fibonacci(x)

    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    for i in range(10):
        assert foo(i) == fib[i]
