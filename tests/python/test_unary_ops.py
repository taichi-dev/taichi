import numpy as np
import pytest
from taichi.lang.exception import TaichiTypeError

import taichi as ti
from tests import test_utils


def _test_op(dt, taichi_op, np_op):
    print("arch={} default_fp={}".format(ti.lang.impl.current_cfg().arch, ti.lang.impl.current_cfg().default_fp))
    n = 4
    val = ti.field(dt, shape=n)

    def f(i):
        return i * 0.1 + 0.4

    @ti.kernel
    def fill():
        for i in range(n):
            val[i] = taichi_op(ti.func(f)(ti.cast(i, dt)))

    fill()

    # check that it is double precision
    for i in range(n):
        if dt == ti.f64:
            assert abs(np_op(float(f(i))) - val[i]) < 1e-15
        else:
            assert (
                abs(np_op(float(f(i))) - val[i]) < 1e-6
                if ti.lang.impl.current_cfg().arch not in (ti.opengl, ti.gles, ti.vulkan)
                else 1e-5
            )


op_pairs = [
    (ti.sin, np.sin),
    (ti.cos, np.cos),
    (ti.asin, np.arcsin),
    (ti.acos, np.arccos),
    (ti.tan, np.tan),
    (ti.tanh, np.tanh),
    (ti.exp, np.exp),
    (ti.log, np.log),
]


@pytest.mark.parametrize("taichi_op,np_op", op_pairs)
@test_utils.test(default_fp=ti.f32)
def test_trig_f32(taichi_op, np_op):
    _test_op(ti.f32, taichi_op, np_op)


@pytest.mark.parametrize("taichi_op,np_op", op_pairs)
@test_utils.test(require=ti.extension.data64, default_fp=ti.f64)
def test_trig_f64(taichi_op, np_op):
    _test_op(ti.f64, taichi_op, np_op)


@test_utils.test()
def test_bit_not_invalid():
    @ti.kernel
    def test(x: ti.f32) -> ti.i32:
        return ~x

    with pytest.raises(TaichiTypeError, match=r"takes integral inputs only"):
        test(1.0)


@test_utils.test()
def test_logic_not_invalid():
    @ti.kernel
    def test(x: ti.f32) -> ti.i32:
        return not x

    with pytest.raises(TaichiTypeError, match=r"takes integral inputs only"):
        test(1.0)


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.opengl, ti.metal])
def test_frexp():
    @ti.kernel
    def get_frac(x: ti.f32) -> ti.f32:
        a, b = ti.frexp(x)
        return a

    assert test_utils.allclose(get_frac(1.4), 0.7)

    @ti.kernel
    def get_exp(x: ti.f32) -> ti.i32:
        a, b = ti.frexp(x)
        return b

    assert get_exp(1.4) == 1


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_popcnt():
    @ti.kernel
    def test_i32(x: ti.int32) -> ti.int32:
        return ti.math.popcnt(x)

    @ti.kernel
    def test_i64(x: ti.int64) -> ti.int32:
        return ti.math.popcnt(x)

    @ti.kernel
    def test_u32(x: ti.uint32) -> ti.int32:
        return ti.math.popcnt(x)

    @ti.kernel
    def test_u64(x: ti.uint64) -> ti.int32:
        return ti.math.popcnt(x)

    assert test_i32(100) == 3
    assert test_i32(1000) == 6
    assert test_i32(10000) == 5
    assert test_i64(100) == 3
    assert test_i64(1000) == 6
    assert test_i64(10000) == 5
    assert test_u32(100) == 3
    assert test_u32(1000) == 6
    assert test_u32(10000) == 5
    assert test_u64(100) == 3
    assert test_u64(1000) == 6
    assert test_i64(10000) == 5


@test_utils.test(arch=[ti.cpu, ti.metal, ti.cuda, ti.vulkan])
def test_clz():
    @ti.kernel
    def test_i32(x: ti.int32) -> ti.int32:
        return ti.math.clz(x)

    # assert test_i32(0) == 32
    assert test_i32(1) == 31
    assert test_i32(2) == 30
    assert test_i32(3) == 30
    assert test_i32(4) == 29
    assert test_i32(5) == 29
    assert test_i32(1023) == 22
    assert test_i32(1024) == 21


@test_utils.test(arch=[ti.metal])
def test_popcnt():
    @ti.kernel
    def test_i32(x: ti.int32) -> ti.int32:
        return ti.math.popcnt(x)

    @ti.kernel
    def test_u32(x: ti.uint32) -> ti.int32:
        return ti.math.popcnt(x)

    assert test_i32(100) == 3
    assert test_i32(1000) == 6
    assert test_i32(10000) == 5
    assert test_u32(100) == 3
    assert test_u32(1000) == 6
    assert test_u32(10000) == 5


@test_utils.test()
def test_sign():
    @ti.kernel
    def foo(val: ti.f32) -> ti.f32:
        return ti.math.sign(val)

    assert foo(0.5) == 1.0
    assert foo(-0.5) == -1.0
    assert foo(0.0) == 0.0
