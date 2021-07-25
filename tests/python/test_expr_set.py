import taichi as ti


@ti.test(ti.cpu)
def test_expr_set_basic():
    @ti.kernel
    def func() -> ti.i32:
        x = {2, 4, 6}
        return len(x)

    assert func() == 3
