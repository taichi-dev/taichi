import taichi as ti


@ti.test(experimental_real_function=True)
def test_function_without_return():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def foo(val: ti.i32) -> ti.i32:
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


@ti.test(experimental_real_function=True)
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
        if ti.cfg.arch != ti.opengl:
            # OpenGL backend does not support assertions
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


@ti.must_throw(AssertionError)
def test_failing_multiple_return():
    ti.init(experimental_real_function=True)
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
