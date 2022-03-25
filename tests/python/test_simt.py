import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda)
def test_shfl_down_i32():
    a = ti.field(dtype=ti.i32, shape=32)
    b = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.lang.shfl_down_i32(ti.u32(0xFFFFFFFF), b[i], 1)

    for i in range(32):
        b[i] = i * i

    foo()

    for i in range(31):
        assert a[i] == b[i + 1]
