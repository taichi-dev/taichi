import taichi as ti


@ti.test(ti.cpu)
def test_named_expr():
    @ti.kernel
    def func() -> ti.i32:
        a = (b := 4) + 1
        return a * 10 + b

    assert func() == 54
