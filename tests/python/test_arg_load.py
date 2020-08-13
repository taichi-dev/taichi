import taichi as ti


@ti.all_archs
def test_arg_load():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32)

    ti.root.place(x, y)

    @ti.kernel
    def set_i32(v: ti.i32):
        x[None] = v

    @ti.kernel
    def set_f32(v: ti.f32):
        y[None] = v

    set_i32(123)
    assert x[None] == 123

    set_i32(456)
    assert x[None] == 456

    set_f32(0.125)
    assert y[None] == 0.125

    set_f32(1.5)
    assert y[None] == 1.5


@ti.require(ti.extension.data64)
@ti.all_archs
def test_arg_load_f64():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32)

    ti.root.place(x, y)

    @ti.kernel
    def set_f64(v: ti.f64):
        y[None] = ti.cast(v, ti.f32)

    @ti.kernel
    def set_i64(v: ti.i64):
        y[None] = v

    set_i64(789)
    assert y[None] == 789

    set_f64(2.5)
    assert y[None] == 2.5


@ti.all_archs
def test_ext_arr():
    N = 128
    x = ti.field(ti.f32)

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def set_f32(v: ti.ext_arr()):
        for i in range(N):
            x[i] = v[i] + i

    import numpy as np
    v = np.ones((N, ), dtype=np.float32) * 10
    set_f32(v)
    for i in range(N):
        assert x[i] == 10 + i
