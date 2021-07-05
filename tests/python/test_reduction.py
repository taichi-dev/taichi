import numpy as np
import pytest
from pytest import approx

import taichi as ti

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
    if ti.cfg.arch == ti.opengl and dtype == ti.f32:
        # OpenGL is not capable of such large number in its float32...
        N = 1024 * 16

    a = ti.field(dtype, shape=N)
    tot = ti.field(dtype, shape=())

    @ti.kernel
    def fill():
        for i in a:
            a[i] = i

    ti_op = ti_ops[op]

    @ti.kernel
    def reduce():
        for i in a:
            ti_op(tot[None], a[i])

    @ti.kernel
    def reduce_tmp() -> dtype:
        s = ti.zero(tot[None])
        for i in a:
            ti_op(s, a[i])
        return s

    fill()
    reduce()
    tot2 = reduce_tmp()

    ground_truth = np_ops[op](a.to_numpy())

    assert criterion(tot[None], ground_truth)
    assert criterion(tot2, ground_truth)


@pytest.mark.parametrize('op', [OP_ADD, OP_MIN, OP_MAX, OP_AND, OP_OR, OP_XOR])
@ti.all_archs
def test_reduction_single_i32(op):
    _test_reduction_single(ti.i32, lambda x, y: x % 2**32 == y % 2**32, op)


@pytest.mark.parametrize('op', [OP_ADD])
@ti.test(exclude=ti.opengl)
def test_reduction_single_u32(op):
    _test_reduction_single(ti.u32, lambda x, y: x % 2**32 == y % 2**32, op)


@pytest.mark.parametrize('op', [OP_ADD, OP_MIN, OP_MAX])
@ti.all_archs
def test_reduction_single_f32(op):
    _test_reduction_single(ti.f32, lambda x, y: x == approx(y, 3e-4), op)


@pytest.mark.parametrize('op', [OP_ADD])
@ti.require(ti.extension.data64)
@ti.all_archs
def test_reduction_single_i64(op):
    _test_reduction_single(ti.i64, lambda x, y: x % 2**64 == y % 2**64, op)


@pytest.mark.parametrize('op', [OP_ADD])
@ti.require(ti.extension.data64)
@ti.archs_excluding(ti.opengl)  # OpenGL doesn't have u64 yet
def test_reduction_single_u64(op):
    _test_reduction_single(ti.u64, lambda x, y: x % 2**64 == y % 2**64, op)


@pytest.mark.parametrize('op', [OP_ADD])
@ti.require(ti.extension.data64)
@ti.all_archs
def test_reduction_single_f64(op):
    _test_reduction_single(ti.f64, lambda x, y: x == approx(y, 1e-12), op)


@ti.all_archs
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
