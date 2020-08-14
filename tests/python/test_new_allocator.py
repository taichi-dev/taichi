import taichi as ti


@ti.all_archs
def test_1d():
    N = 16

    x = ti.field(ti.f32, shape=(N, ))
    y = ti.field(ti.f32, shape=(N, ))

    @ti.kernel
    def func():
        for i in range(N):
            y[i] = x[i]

    for i in range(N):
        x[i] = i * 2

    func()

    for i in range(N):
        assert y[i] == i * 2


@ti.all_archs
def test_3d():
    N = 2
    M = 2

    x = ti.field(ti.f32, shape=(N, M))
    y = ti.field(ti.f32, shape=(N, M))

    @ti.kernel
    def func():
        for I in ti.grouped(x):
            y[I] = x[I]

    for i in range(N):
        for j in range(M):
            x[i, j] = i * 10 + j

    func()

    for i in range(N):
        for j in range(M):
            assert y[i, j] == i * 10 + j


@ti.all_archs
def test_matrix():
    N = 16

    x = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(N, ), layout=ti.AOS)

    @ti.kernel
    def func():
        for i in range(N):
            x[i][1, 1] = x[i][0, 0]

    for i in range(N):
        x[i][0, 0] = i + 3

    func()

    for i in range(N):
        assert x[i][1, 1] == i + 3


@ti.all_archs
def test_alloc_in_kernel():
    return  # build bots may not have this much memory to tests...
    x = ti.field(ti.f32)

    ti.root.pointer(ti.i, 8192).dense(ti.i, 1024 * 1024).place(x)

    @ti.kernel
    def touch():
        for i in range(4096):
            x[i * 1024 * 1024] = 1

    touch()
