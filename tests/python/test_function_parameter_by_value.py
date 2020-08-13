import taichi as ti


@ti.all_archs
def test_pass_by_value():
    @ti.func
    def set_val(x, i):
        x = i

    ret = ti.field(ti.i32, shape=())

    @ti.kernel
    def task():
        set_val(ret[None], 112)

    task()
    assert ret[None] == 0
