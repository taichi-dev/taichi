import taichi as ti


@ti.all_archs
def test_loops():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    N = 512

    ti.root.dense(ti.i, N).place(x)
    ti.root.dense(ti.i, N).place(y)
    ti.root.lazy_grad()

    for i in range(N // 2, N):
        y[i] = i - 300

    @ti.kernel
    def func():
        for i in range(ti.static(N // 2 + 3), N):
            x[i] = ti.abs(y[i])

    func()

    for i in range(N // 2 + 3):
        assert x[i] == 0

    for i in range(N // 2 + 3, N):
        assert x[i] == abs(y[i])


@ti.all_archs
def test_numpy_loops():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    N = 512

    ti.root.dense(ti.i, N).place(x)
    ti.root.dense(ti.i, N).place(y)
    ti.root.lazy_grad()

    for i in range(N // 2, N):
        y[i] = i - 300

    import numpy as np
    begin = (np.ones(1) * (N // 2 + 3)).astype(np.int32)
    end = (np.ones(1) * N).astype(np.int32)

    @ti.kernel
    def func():
        for i in range(begin, end):
            x[i] = ti.abs(y[i])

    func()

    for i in range(N // 2 + 3):
        assert x[i] == 0

    for i in range(N // 2 + 3, N):
        assert x[i] == abs(y[i])


@ti.all_archs
def test_nested_loops():
    # this may crash if any LLVM allocas are called in the loop body
    x = ti.field(ti.i32)

    n = 2048

    ti.root.dense(ti.ij, n).place(x)

    @ti.kernel
    def paint():
        for i in range(n):
            for j in range(n):
                x[0, 0] = i

    paint()


@ti.all_archs
def test_zero_outer_loop():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def test():
        for i in range(0):
            x[None] = 1

    test()

    assert x[None] == 0


@ti.all_archs
def test_zero_inner_loop():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def test():
        for i in range(1):
            for j in range(0):
                x[None] = 1

    test()

    assert x[None] == 0


@ti.all_archs
def test_dynamic_loop_range():
    x = ti.field(ti.i32)
    c = ti.field(ti.i32)
    n = 2000

    ti.root.dense(ti.i, n).place(x)
    ti.root.place(c)

    @ti.kernel
    def test():
        for i in x:
            x[i] = ti.atomic_add(c[None], 1)
        for i in range(c[None], c[None] * 2):
            x[i - n] += c[None]

    test()
    assert c[None] == n
    assert sum(x.to_numpy()) == (n * (n - 1) // 2) + n * n


@ti.all_archs
def test_loop_arg_as_range():
    # Dynamic range loops are intended to make sure global tmps work
    x = ti.field(ti.i32)
    n = 1000

    ti.root.dense(ti.i, n).place(x)

    @ti.kernel
    def test(b: ti.i32, e: ti.i32):
        for i in range(b, e):
            x[i - b] = i

    pairs = [
        (0, n // 2),
        (n // 2, n),
        (-n // 2, -n // 3),
    ]
    for b, e in pairs:
        test(b, e)
        for i in range(b, e):
            assert x[i - b] == i


@ti.all_archs
def test_assignment_in_nested_loops():
    # https://github.com/taichi-dev/taichi/issues/1109
    m = ti.field(ti.f32, 3)
    x = ti.field(ti.f32, ())

    @ti.kernel
    def func():
        a = x[None]
        for i in m:
            b = a
            for j in range(1):
                b = b
            x[None] = b

    x[None] = 1
    func()
    assert x[None] == 1
