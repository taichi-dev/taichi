import taichi as ti


@ti.all_archs
def test_function_argument_pass_by_value():
    @ti.func
    def set_val(x, i):
        x = i

    ret = ti.var(ti.i32, shape=())

    @ti.kernel
    def task():
        set_val(ret[None], 112)

    task()
    assert ret[None] == 0


@ti.all_archs
def test_kernel_argument_pass_by_value():
    @ti.kernel
    def task(x1: ti.i32, x2: ti.i32):
        x1 = x1
        x2 = x2
        if x1 > x2:
            x1 = x2
        ret[None] = x1

    ret = ti.var(ti.i32, shape=())

    task(15, 13)
    assert ret[None] == 13
    task(15, 18)
    assert ret[None] == 15
