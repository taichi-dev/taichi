import pytest

import taichi as ti


@ti.test(experimental_real_function=True)
def test_function_without_return():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def foo(val: ti.i32):
        x[None] += val

    @ti.kernel
    def run():
        foo(40)
        foo(2)

    x[None] = 0
    run()
    assert x[None] == 42


@ti.test(experimental_real_function=True)
def test_function_with_return():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def foo(val: ti.i32) -> ti.i32:
        x[None] += val
        return val

    @ti.kernel
    def run():
        a = foo(40)
        foo(2)
        assert a == 40

    x[None] = 0
    run()
    assert x[None] == 42


@ti.test(experimental_real_function=True, exclude=[ti.opengl, ti.cc])
def test_function_with_multiple_last_return():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def foo(val: ti.i32) -> ti.i32:
        if x[None]:
            x[None] += val * 2
            return val * 2
        else:
            x[None] += val
            return val

    @ti.kernel
    def run():
        a = foo(40)
        foo(1)
        assert a == 40

    x[None] = 0
    run()
    assert x[None] == 42


@ti.test(experimental_real_function=True)
def test_call_expressions():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def foo(val: ti.i32) -> ti.i32:
        if x[None] > 10:
            x[None] += 1
        x[None] += val
        return 0

    @ti.kernel
    def run():
        assert foo(15) == 0
        assert foo(10) == 0

    x[None] = 0
    run()
    assert x[None] == 26


@ti.test(arch=ti.cpu, experimental_real_function=True)
@ti.must_throw(AssertionError)
def test_failing_multiple_return():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def foo(val: ti.i32) -> ti.i32:
        if x[None] > 10:
            if x[None] > 20:
                return 1
            x[None] += 1
        x[None] += val
        return 0

    @ti.kernel
    def run():
        assert foo(15) == 0
        assert foo(10) == 0
        assert foo(100) == 1

    x[None] = 0
    run()
    assert x[None] == 26


@ti.test(experimental_real_function=True)
def test_python_function():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def inc(val: ti.i32):
        x[None] += val

    def identity(x):
        return x

    @ti.data_oriented
    class A:
        def __init__(self):
            self.count = ti.field(ti.i32, shape=())
            self.count[None] = 0

        @ti.pyfunc
        def dec(self, val: ti.i32) -> ti.i32:
            self.count[None] += 1
            x[None] -= val
            return self.count[None]

        @ti.kernel
        def run(self) -> ti.i32:
            a = self.dec(1)
            identity(2)
            inc(identity(3))
            return a

    a = A()
    x[None] = 0
    assert a.run() == 1
    assert a.run() == 2
    assert x[None] == 4
    assert a.dec(4) == 3
    assert x[None] == 0


@ti.test(arch=[ti.cpu, ti.cuda], debug=True)
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


@ti.test(experimental_real_function=True)
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

    @ti.func
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


@ti.test(experimental_real_function=True)
def test_missing_arg_annotation():
    with pytest.raises(ti.KernelDefError, match='must be type annotated'):

        @ti.func
        def add(a, b: ti.i32) -> ti.i32:
            return a + b


@ti.test(experimental_real_function=True)
def test_missing_return_annotation():
    with pytest.raises(ti.TaichiSyntaxError,
                       match='return value must be annotated'):

        @ti.func
        def add(a: ti.i32, b: ti.i32):
            return a + b

        @ti.kernel
        def run():
            add(30, 2)

        run()
