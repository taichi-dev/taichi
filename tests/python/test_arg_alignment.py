import taichi as ti


@ti.test(exclude=[ti.opengl])
def test_ret_write():

    @ti.kernel
    def func(a: ti.i16) -> ti.f32:
        return 3.0

    assert func(255) == 3.0


@ti.test(exclude=[ti.opengl])
def test_arg_read():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def func(a: ti.i8, b: ti.i32):
        x[None] = b

    func(255, 2)
    assert x[None] == 2
