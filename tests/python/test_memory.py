import taichi as ti


@ti.test(arch=ti.cuda, use_unified_memory=True)
def test_unified_memory_allocate():
    HUGE_SIZE = 1024**3
    x = ti.field(ti.i32, shape=(HUGE_SIZE, ))
    for i in range(10):
        x[i] = i
    # There is no checking for this. If UM is not implemented correctly, this
    # test would crash.
