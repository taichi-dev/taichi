import numpy as np
from pytest import approx
from taichi.lang.simt import subgroup

import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda)
def test_all_nonzero():
    a = ti.field(dtype=ti.i32, shape=32)
    b = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.all_nonzero(ti.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 1
        a[i] = -1

    foo()

    for i in range(32):
        assert a[i] == 1

    b[np.random.randint(0, 32)] = 0

    foo()

    for i in range(32):
        assert a[i] == 0


@test_utils.test(arch=ti.cuda)
def test_any_nonzero():
    a = ti.field(dtype=ti.i32, shape=32)
    b = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.any_nonzero(ti.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 0
        a[i] = -1

    foo()

    for i in range(32):
        assert a[i] == 0

    b[np.random.randint(0, 32)] = 1

    foo()

    for i in range(32):
        assert a[i] == 1


@test_utils.test(arch=ti.cuda)
def test_unique():
    a = ti.field(dtype=ti.u32, shape=32)
    b = ti.field(dtype=ti.u32, shape=32)

    @ti.kernel
    def check():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.unique(ti.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 0
        a[i] = -1

    check()

    for i in range(32):
        assert a[i] == 1

    for i in range(32):
        b[i] = i + 100

    check()

    for i in range(32):
        assert a[i] == 1

    b[np.random.randint(0, 32)] = 0

    check()

    for i in range(32):
        assert a[i] == 0


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
        b[i] = i % 2
        key += b[i] * pow(2, i)

    foo()

    for i in range(32):
        assert a[i] == key


@test_utils.test(arch=ti.cuda)
def test_shfl_sync_i32():
    a = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.shfl_sync_i32(ti.u32(0xFFFFFFFF), a[i], 0)

    for i in range(32):
        a[i] = i + 1

    foo()

    for i in range(1, 32):
        assert a[i] == 1


@test_utils.test(arch=ti.cuda)
def test_shfl_sync_f32():
    a = ti.field(dtype=ti.f32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.shfl_sync_f32(ti.u32(0xFFFFFFFF), a[i], 0)

    for i in range(32):
        a[i] = i + 1.0

    foo()

    for i in range(1, 32):
        assert a[i] == approx(1.0, abs=1e-4)


@test_utils.test(arch=ti.cuda)
def test_shfl_up_i32():
    # TODO
    pass


@test_utils.test(arch=ti.cuda)
def test_shfl_xor_i32():
    a = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            for j in range(5):
                offset = 1 << j
                a[i] += ti.simt.warp.shfl_xor_i32(ti.u32(0xFFFFFFFF), a[i],
                                                  offset)

    value = 0
    for i in range(32):
        a[i] = i
        value += i

    foo()

    for i in range(32):
        assert a[i] == value


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
def test_shfl_down_f32():
    a = ti.field(dtype=ti.f32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = ti.simt.warp.shfl_down_f32(ti.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i * 0.9

    foo()

    for i in range(31):
        assert a[i] == approx((i + 1) * (i + 1) * 0.9, abs=1e-4)


@test_utils.test(arch=ti.cuda)
def test_match_any():
    a = ti.field(dtype=ti.i32, shape=32)
    b = ti.field(dtype=ti.u32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(16):
            a[i] = 0
            a[i + 16] = 1

        for i in range(32):
            b[i] = ti.simt.warp.match_any(ti.u32(0xFFFFFFFF), a[i])

    foo()

    for i in range(16):
        assert b[i] == 65535
    for i in range(16):
        assert b[i + 16] == (2**32 - 2**16)


@test_utils.test(arch=ti.cuda)
def test_match_all():
    a = ti.field(dtype=ti.i32, shape=32)
    b = ti.field(dtype=ti.u32, shape=32)
    c = ti.field(dtype=ti.u32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = 1
        for i in range(32):
            b[i] = ti.simt.warp.match_all(ti.u32(0xFFFFFFFF), a[i])

        a[0] = 2
        for i in range(32):
            c[i] = ti.simt.warp.match_all(ti.u32(0xFFFFFFFF), a[i])

    foo()

    for i in range(32):
        assert b[i] == (2**32 - 1)

    for i in range(32):
        assert c[i] == 0


@test_utils.test(arch=ti.cuda)
def test_active_mask():
    a = ti.field(dtype=ti.u32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=16)
        for i in range(32):
            a[i] = ti.simt.warp.active_mask()

    foo()

    for i in range(32):
        assert a[i] == 65535


@test_utils.test(arch=ti.cuda)
def test_warp_sync():
    a = ti.field(dtype=ti.u32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            a[i] = i
        ti.simt.warp.sync(ti.u32(0xFFFFFFFF))
        for i in range(16):
            a[i] = a[i + 16]

    foo()

    for i in range(32):
        assert a[i] == i % 16 + 16


@test_utils.test(arch=ti.cuda)
def test_block_sync():
    N = 1024
    a = ti.field(dtype=ti.u32, shape=N)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=N)
        for i in range(N):
            # Make the 0-th thread runs slower intentionally
            for j in range(N - i):
                a[i] = j
            ti.simt.block.sync()
            if i > 0:
                a[i] = a[0]

    foo()

    for i in range(N):
        assert a[i] == N - 1


# TODO: replace this with a stronger test case
@test_utils.test(arch=ti.cuda)
def test_grid_memfence():

    N = 1000
    BLOCK_SIZE = 1
    a = ti.field(dtype=ti.u32, shape=N)

    @ti.kernel
    def foo():

        block_counter = 0
        ti.loop_config(block_dim=BLOCK_SIZE)
        for i in range(N):

            a[i] = 1
            ti.simt.grid.memfence()

            # Execute a prefix sum after all blocks finish
            actual_order_of_block = ti.atomic_add(block_counter, 1)
            if actual_order_of_block == N - 1:
                for j in range(1, N):
                    a[j] += a[j - 1]

    foo()

    for i in range(N):
        assert a[i] == i + 1


# Higher level primitives test
def _test_subgroup_reduce(op, group_op, np_op, size, initial_value, dtype):
    field = ti.field(dtype, (size))
    if dtype == ti.i32 or dtype == ti.i64:
        rand_values = np.random.randint(1, 100, size=(size))
        field.from_numpy(rand_values)
    if dtype == ti.f32 or dtype == ti.f64:
        rand_values = np.random.random(size=(size)).astype(np.float32)
        field.from_numpy(rand_values)

    @ti.kernel
    def reduce_all() -> dtype:
        sum = ti.cast(initial_value, dtype)
        for i in field:
            value = field[i]
            reduce_value = group_op(value)
            if subgroup.elect():
                op(sum, reduce_value)
        return sum

    if dtype == ti.i32 or dtype == ti.i64:
        assert (reduce_all() == np_op(rand_values))
    else:
        assert (reduce_all() == approx(np_op(rand_values), 3e-4))


# We use 2677 as size because it is a prime number
# i.e. any device other than a subgroup size of 1 should have one non active group


@test_utils.test(arch=ti.vulkan)
def test_subgroup_reduction_add_i32():
    _test_subgroup_reduce(ti.atomic_add, subgroup.reduce_add, np.sum, 2677, 0,
                          ti.i32)


@test_utils.test(arch=ti.vulkan)
def test_subgroup_reduction_add_f32():
    _test_subgroup_reduce(ti.atomic_add, subgroup.reduce_add, np.sum, 2677, 0,
                          ti.f32)


# @test_utils.test(arch=ti.vulkan)
# def test_subgroup_reduction_mul_i32():
#     _test_subgroup_reduce(ti.atomic_add, subgroup.reduce_mul, np.prod, 8, 1, ti.f32)


@test_utils.test(arch=ti.vulkan)
def test_subgroup_reduction_max_i32():
    _test_subgroup_reduce(ti.atomic_max, subgroup.reduce_max, np.max, 2677, 0,
                          ti.i32)


@test_utils.test(arch=ti.vulkan)
def test_subgroup_reduction_max_f32():
    _test_subgroup_reduce(ti.atomic_max, subgroup.reduce_max, np.max, 2677, 0,
                          ti.f32)


@test_utils.test(arch=ti.vulkan)
def test_subgroup_reduction_min_f32():
    _test_subgroup_reduce(ti.atomic_max, subgroup.reduce_max, np.max, 2677, 0,
                          ti.f32)
