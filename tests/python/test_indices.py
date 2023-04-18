import pytest
from taichi.lang.misc import get_host_arch_list

import taichi as ti
from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_indices():
    a = ti.field(ti.f32, shape=(128, 32, 8))

    b = ti.field(ti.f32)
    ti.root.dense(ti.j, 32).dense(ti.i, 16).place(b)

    mapping_a = a.snode._physical_index_position()

    assert mapping_a == {0: 0, 1: 1, 2: 2}

    mapping_b = b.snode._physical_index_position()

    assert mapping_b == {0: 0, 1: 1}
    # Note that b is column-major:
    # the virtual first index exposed to the user comes second in memory layout.

    @ti.kernel
    def fill():
        for i, j in b:
            b[i, j] = i * 10 + j

    @ti.kernel
    def get_field_addr(i: ti.i32, j: ti.i32) -> ti.u64:
        return ti.get_addr(b, [i, j])

    fill()
    for i in range(16):
        for j in range(32):
            assert b[i, j] == i * 10 + j
    assert get_field_addr(0, 1) + 4 == get_field_addr(1, 1)


@test_utils.test(arch=get_host_arch_list(), default_ip=ti.i64)
def test_indices_i64():
    n = 1024
    val = ti.field(dtype=ti.i64, shape=n)
    val.fill(1)

    @ti.kernel
    def prefix_sum():
        ti.loop_config(serialize=True)
        for i in range(1, 1024):
            val[i] += val[i - 1]

    prefix_sum()
    for i in range(n):
        assert val[i] == i + 1


@test_utils.test()
def test_indices_with_matrix():
    grid_m = ti.field(dtype=ti.i32, shape=(10, 10))

    @ti.kernel
    def build_grid():
        base = int(ti.Vector([2, 4]))
        grid_m[base] = 100

        grid_m[int(ti.Vector([1, 1]))] = 10

    build_grid()

    assert grid_m[1, 1] == 10
    assert grid_m[2, 4] == 100


@test_utils.test()
def test_negative_valued_indices():
    @ti.kernel
    def foo(i: int):
        x = ti.Vector([i, i + 1, i + 2])
        print(x[:-1])

    with pytest.raises(
        ti.TaichiSyntaxError,
        match="Negative indices are not supported in Taichi kernels.",
    ):
        foo(0)
