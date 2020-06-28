import taichi as ti


# The first test to run, ever:
def test_000_without_init():
    assert ti.cfg.arch == ti.cpu

    x = ti.var(ti.i32, (2, 3))
    assert x.shape == (2, 3)

    x[1, 2] = 4
    assert x[1, 2] == 4


@ti.all_archs
@ti.all_archs
@ti.must_throw(RuntimeError)
def test_materialization_after_kernel():
    x = ti.var(ti.f32, (3, 4))

    @ti.kernel
    def func():
        print(x[2, 3])

    func()

    y = ti.var(ti.f32, (2, 3))
    # ERROR: No new variable should be declared after kernel invocation!


@ti.all_archs
@ti.must_throw(RuntimeError)
def test_materialization_after_access():
    x = ti.var(ti.f32, (3, 4))

    print(x[2, 3])

    y = ti.var(ti.f32, (2, 3))
    # ERROR: No new variable should be declared after Python-scope tensor access!


@ti.all_archs
@ti.must_throw(RuntimeError)
def test_materialization_after_get_shape():
    x = ti.var(ti.f32, (3, 4))

    print(x.shape)

    y = ti.var(ti.f32, (2, 3))
    # ERROR: No new variable should be declared after Python-scope tensor access!
