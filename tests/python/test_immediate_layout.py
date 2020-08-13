import taichi as ti


@ti.all_archs
def test_1D():
    N = 2
    x = ti.field(ti.f32)
    ti.root.dense(ti.i, N).place(x)

    x[0] = 42
    assert x[0] == 42
    assert x[1] == 0
