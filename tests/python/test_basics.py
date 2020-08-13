import taichi as ti


@ti.all_archs
def test_simple():
    n = 128
    x = ti.field(ti.i32, shape=n)

    @ti.kernel
    def func():
        x[7] = 120

    func()

    for i in range(n):
        if i == 7:
            assert x[i] == 120
        else:
            assert x[i] == 0


@ti.all_archs
def test_range_loops():
    n = 128
    x = ti.field(ti.i32, shape=n)

    @ti.kernel
    def func():
        for i in range(n):
            x[i] = i + 123

    func()

    for i in range(n):
        assert x[i] == i + 123


@ti.all_archs
def test_python_access():
    n = 128
    x = ti.field(ti.i32, shape=n)

    x[3] = 123
    x[4] = 456
    assert x[3] == 123
    assert x[4] == 456


@ti.all_archs
def test_if():
    x = ti.field(ti.f32, shape=16)

    @ti.kernel
    def if_test():
        for i in x:
            if i < 100:
                x[i] = 100
            else:
                x[i] = i

    if_test()

    for i in range(16):
        assert x[i] == 100

    @ti.kernel
    def if_test2():
        for i in x:
            if i < 100:
                x[i] = i
            else:
                x[i] = 100

    if_test2()

    for i in range(16):
        assert x[i] == i


@ti.all_archs
def test_if_global_load():
    x = ti.field(ti.i32, shape=16)

    @ti.kernel
    def fill():
        for i in x:
            if x[i]:
                x[i] = i

    for i in range(16):
        x[i] = i % 2

    fill()

    for i in range(16):
        if i % 2 == 0:
            assert x[i] == 0
        else:
            assert x[i] == i


@ti.all_archs
def test_while_global_load():
    x = ti.field(ti.i32, shape=16)
    y = ti.field(ti.i32, shape=())

    @ti.kernel
    def run():
        while x[3]:
            x[3] -= 1
            y[None] += 1

    for i in range(16):
        x[i] = i

    run()

    assert y[None] == 3
