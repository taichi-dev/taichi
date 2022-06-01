import numpy as np
import pytest
from pytest import approx

import taichi as ti
from tests import test_utils

OP_ADD = 0
OP_MIN = 1
OP_MAX = 2
OP_AND = 3
OP_OR = 4
OP_XOR = 5

ti_ops = {
    OP_ADD: ti.atomic_add,
    OP_MIN: ti.atomic_min,
    OP_MAX: ti.atomic_max,
    OP_AND: ti.atomic_and,
    OP_OR: ti.atomic_or,
    OP_XOR: ti.atomic_xor
}

np_ops = {
    OP_ADD: np.sum,
    OP_MIN: lambda a: a.min(),
    OP_MAX: lambda a: a.max(),
    OP_AND: np.bitwise_and.reduce,
    OP_OR: np.bitwise_or.reduce,
    OP_XOR: np.bitwise_xor.reduce
}


def _test_reduction_single(dtype, criterion, op):
    N = 1024 * 1024
    if (ti.lang.impl.current_cfg().arch == ti.opengl
            or ti.lang.impl.current_cfg().arch == ti.vulkan
            or ti.lang.impl.current_cfg().arch == ti.dx11) and dtype == ti.f32:
        # OpenGL/Vulkan are not capable of such large number in its float32...
        N = 1024 * 16

    a = ti.field(dtype, shape=N)
    tot = ti.field(dtype, shape=())

    if dtype in [ti.f32, ti.f64]:

        @ti.kernel
        def fill():
            for i in a:
                a[i] = i + 0.5
    else:

        @ti.kernel
        def fill():
            for i in a:
                a[i] = i + 1

    ti_op = ti_ops[op]

    @ti.kernel
    def reduce():
        for i in a:
            ti_op(tot[None], a[i])

    @ti.kernel
    def reduce_tmp() -> dtype:
        s = ti.zero(tot[None]) if op == OP_ADD or op == OP_XOR else a[0]
        for i in a:
            ti_op(s, a[i])
        return s

    fill()
    tot[None] = 0 if op in [OP_ADD, OP_XOR] else a[0]
    reduce()
    tot2 = reduce_tmp()

    np_arr = a.to_numpy()
    ground_truth = np_ops[op](np_arr)

    assert criterion(tot[None], ground_truth)
    assert criterion(tot2, ground_truth)


@pytest.mark.parametrize('op', [OP_ADD, OP_MIN, OP_MAX, OP_AND, OP_OR, OP_XOR])
@test_utils.test()
def test_reduction_single_i32(op):
    _test_reduction_single(ti.i32, lambda x, y: x % 2**32 == y % 2**32, op)


@pytest.mark.parametrize('op', [OP_ADD])
@test_utils.test(exclude=ti.opengl)
def test_reduction_single_u32(op):
    _test_reduction_single(ti.u32, lambda x, y: x % 2**32 == y % 2**32, op)


@pytest.mark.parametrize('op', [OP_ADD, OP_MIN, OP_MAX])
@test_utils.test()
def test_reduction_single_f32(op):
    _test_reduction_single(ti.f32, lambda x, y: x == approx(y, 3e-4), op)


@pytest.mark.parametrize('op', [OP_ADD])
@test_utils.test(require=ti.extension.data64)
def test_reduction_single_i64(op):
    _test_reduction_single(ti.i64, lambda x, y: x % 2**64 == y % 2**64, op)


@pytest.mark.parametrize('op', [OP_ADD])
@test_utils.test(exclude=ti.opengl, require=ti.extension.data64)
def test_reduction_single_u64(op):
    _test_reduction_single(ti.u64, lambda x, y: x % 2**64 == y % 2**64, op)


@pytest.mark.parametrize('op', [OP_ADD])
@test_utils.test(require=ti.extension.data64)
def test_reduction_single_f64(op):
    _test_reduction_single(ti.f64, lambda x, y: x == approx(y, 1e-12), op)


@test_utils.test()
def test_reduction_different_scale():
    @ti.kernel
    def func(n: ti.template()) -> ti.i32:
        x = 0
        for i in range(n):
            ti.atomic_add(x, 1)
        return x

    # 10 and 60 since OpenGL TLS stride size = 32
    # 1024 and 100000 since OpenGL max threads per group ~= 1792
    for n in [1, 10, 60, 1024, 100000]:
        assert n == func(n)


@test_utils.test()
def test_reduction_ndarray():
    @ti.kernel
    def reduce(a: ti.types.ndarray()) -> ti.i32:
        s = 0
        for i in a:
            ti.atomic_add(s, a[i])
            ti.atomic_sub(s, 2)
        return s

    n = 1024
    x = np.ones(n, dtype=np.int32)
    assert reduce(x) == -n
