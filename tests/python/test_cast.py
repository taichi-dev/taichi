import taichi as ti


@ti.all_archs
def test_cast_f32():
    z = ti.var(ti.i32, shape=())

    @ti.kernel
    def func():
        z[None] = ti.cast(1e9, ti.f32) / ti.cast(1e6, ti.f32) + 1e-3

    func()
    assert z[None] == 1000


@ti.require(ti.extension.data64)
@ti.all_archs
def test_cast_f64():
    z = ti.var(ti.i32, shape=())

    @ti.kernel
    def func():
        z[None] = ti.cast(1e13, ti.f64) / ti.cast(1e10, ti.f64) + 1e-3

    func()
    assert z[None] == 1000


@ti.all_archs
def test_cast_within_while():
    ret = ti.var(ti.i32, shape=())

    @ti.kernel
    def func():
        t = 10
        while t > 5:
            t = 1.0
            break
        ret[None] = t

    func()
