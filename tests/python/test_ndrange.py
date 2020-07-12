import taichi as ti


@ti.all_archs
def test_1d():
    x = ti.var(ti.f32, shape=(16))

    @ti.kernel
    def func():
        for i in ti.ndrange((4, 10)):
            x[i] = i

    func()

    for i in range(16):
        if 4 <= i < 10:
            assert x[i] == i
        else:
            assert x[i] == 0


@ti.all_archs
def test_2d():
    x = ti.var(ti.f32, shape=(16, 32))

    t = 8

    @ti.kernel
    def func():
        for i, j in ti.ndrange((4, 10), (3, t)):
            val = i + j * 10
            x[i, j] = val

    func()
    for i in range(16):
        for j in range(32):
            if 4 <= i < 10 and 3 <= j < 8:
                assert x[i, j] == i + j * 10
            else:
                assert x[i, j] == 0


@ti.all_archs
def test_3d():
    x = ti.var(ti.f32, shape=(16, 32, 64))

    @ti.kernel
    def func():
        for i, j, k in ti.ndrange((4, 10), (3, 8), 17):
            x[i, j, k] = i + j * 10 + k * 100

    func()
    for i in range(16):
        for j in range(32):
            for k in range(64):
                if 4 <= i < 10 and 3 <= j < 8 and k < 17:
                    assert x[i, j, k] == i + j * 10 + k * 100
                else:
                    assert x[i, j, k] == 0


@ti.all_archs
def test_static_grouped():
    x = ti.var(ti.f32, shape=(16, 32, 64))

    @ti.kernel
    def func():
        for I in ti.static(ti.grouped(ti.ndrange((4, 5), (3, 5), 5))):
            x[I] = I[0] + I[1] * 10 + I[2] * 100

    func()
    for i in range(16):
        for j in range(32):
            for k in range(64):
                if 4 <= i < 5 and 3 <= j < 5 and k < 5:
                    assert x[i, j, k] == i + j * 10 + k * 100
                else:
                    assert x[i, j, k] == 0


@ti.all_archs
def test_static_grouped_static():
    x = ti.Matrix(2, 3, dt=ti.f32, shape=(16, 4))

    @ti.kernel
    def func():
        for i, j in ti.ndrange(16, 4):
            for I in ti.static(ti.grouped(ti.ndrange(2, 3))):
                x[i, j][I] = I[0] + I[1] * 10 + i + j * 4

    func()
    for i in range(16):
        for j in range(4):
            for k in range(2):
                for l in range(3):
                    assert x[i, j][k, l] == k + l * 10 + i + j * 4
