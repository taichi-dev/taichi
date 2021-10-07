import taichi as ti


@ti.test(arch=ti.cuda)
def test_memory_allocate():
    HUGE_SIZE = 1024**2 * 128
    x = ti.field(ti.i32, shape=(HUGE_SIZE, ))
    for i in range(10):
        x[i] = i
