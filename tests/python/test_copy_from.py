import taichi as ti


@ti.all_archs
def test_scalar():
    n = 16

    x = ti.field(ti.i32, shape=n)
    y = ti.field(ti.i32, shape=n)

    x[1] = 2

    y[0] = 1
    y[2] = 3

    x.copy_from(y)

    assert x[0] == 1
    assert x[1] == 0
    assert x[2] == 3

    assert y[0] == 1
    assert y[1] == 0
    assert y[2] == 3
