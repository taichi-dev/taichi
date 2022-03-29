from pytest import approx

import taichi as ti
from tests import test_utils
import random

@test_utils.test(arch=ti.cuda)
def test_all_nonzero():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_any_nonzero():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_unique():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_ballot():
    a = ti.field(dtype=ti.u32, shape=32)
    b = ti.field(dtype=ti.u32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.ballot(b[i])

    key = 0
    for i in range(32):
        b[i] = random.randint(1, 100) % 2
        key += b[i] * pow(2, i)

    foo()

    for i in range(32):
        assert a[i] == key


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
            a[i] = ti.simt.warp.shfl_down_i32(ti.u32(0xFFFFFFFF), b[i], 1)

    for i in range(32):
        b[i] = i * i

    foo()

    for i in range(31):
        assert a[i] == b[i + 1]

    # TODO: make this test case stronger


@test_utils.test(arch=ti.cuda)
def test_shfl_up_i32():
    a = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.shfl_up_i32(ti.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i

    foo()

    for i in range(1, 32):
        assert a[i] == (i - 1) * (i - 1)


@test_utils.test(arch=ti.cuda)
def test_shfl_up_f32():
    a = ti.field(dtype=ti.f32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.shfl_up_f32(ti.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i * 0.9

    foo()

    for i in range(1, 32):
        assert a[i] == approx((i - 1) * (i - 1) * 0.9, abs=1e-4)


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
