import taichi as ti


@ti.all_archs
def test_clear_all_gradients():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    z = ti.field(ti.f32)
    w = ti.field(ti.f32)

    n = 128

    ti.root.place(x)
    ti.root.dense(ti.i, n).place(y)
    ti.root.dense(ti.i, n).dense(ti.j, n).place(z, w)
    ti.root.lazy_grad()

    x.grad[None] = 3
    for i in range(n):
        y.grad[i] = 3
        for j in range(n):
            z.grad[i, j] = 5
            w.grad[i, j] = 6

    ti.clear_all_gradients()
    assert ti.get_runtime().get_num_compiled_functions() == 3

    assert x.grad[None] == 0
    for i in range(n):
        assert y.grad[i] == 0
        for j in range(n):
            assert z.grad[i, j] == 0
            assert w.grad[i, j] == 0

    ti.clear_all_gradients()
    # No more kernel compilation
    assert ti.get_runtime().get_num_compiled_functions() == 3
