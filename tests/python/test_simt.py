import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda)
def test_all():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_any():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_uni():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_ballot():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_shfl_i32():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_shfl_up_i32():
    # TODO
    pass


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


@test_utils.test(arch=ti.cuda)
def test_match_any():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_match_all():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_active_mask():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_sync():
    # TODO
    pass
