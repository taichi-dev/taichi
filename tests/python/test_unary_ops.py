import numpy as np
import pytest
from taichi.lang.exception import TaichiTypeError

import taichi as ti
from tests import test_utils


def _test_op(dt, taichi_op, np_op):
    print('arch={} default_fp={}'.format(
        ti.lang.impl.current_cfg().arch,
        ti.lang.impl.current_cfg().default_fp))
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
            assert abs(np_op(float(f(i))) -
                       val[i]) < 1e-6 if ti.lang.impl.current_cfg(
                       ).arch not in (ti.opengl, ti.gles, ti.vulkan) else 1e-5


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

    with pytest.raises(TaichiTypeError, match=r'takes integral inputs only'):
        test(1.0)


@test_utils.test()
def test_logic_not_invalid():
    @ti.kernel
    def test(x: ti.f32) -> ti.i32:
        return not x

    with pytest.raises(TaichiTypeError, match=r'takes integral inputs only'):
        test(1.0)


@test_utils.test(arch=ti.cuda)
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
