import math

import pytest
from pytest import approx

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize('use_quant,use_exponent,use_shared_exp',
                         [(False, False, False), (True, False, False),
                          (True, True, False), (True, True, True)])
@test_utils.test(require=ti.extension.quant)
def test_quant_time_integration(use_quant, use_exponent, use_shared_exp):
    if use_quant:
        if use_exponent:
            qflt = ti.types.quant.float(exp=6, frac=13)
            x = ti.Vector.field(2, dtype=qflt)
            if use_shared_exp:
                bitpack = ti.BitpackedFields(max_num_bits=32)
                bitpack.place(x, shared_exponent=True)
                ti.root.place(bitpack)
            else:
                bitpack = ti.BitpackedFields(max_num_bits=32)
                bitpack.place(x.get_scalar_field(0))
                ti.root.place(bitpack)
                bitpack = ti.BitpackedFields(max_num_bits=32)
                bitpack.place(x.get_scalar_field(1))
                ti.root.place(bitpack)
        else:
            qfxt = ti.types.quant.fixed(bits=16, max_value=2)
            x = ti.Vector.field(2, dtype=qfxt)
            bitpack = ti.BitpackedFields(max_num_bits=32)
            bitpack.place(x)
            ti.root.place(bitpack)
    else:
        x = ti.Vector.field(2, dtype=ti.f32, shape=())

    @ti.func
    def v_at(p):
        return ti.Vector([-p[1], p[0]])

    @ti.kernel
    def advance(dt: ti.f32):
        v_mid = v_at(x[None] + 0.5 * dt * v_at(x[None]))
        x[None] = x[None] + v_mid * dt

    x[None] = [1, 0]
    num_steps = 1000
    dt = math.pi * 2 / num_steps
    px = []
    py = []

    N = 1

    for i in range(num_steps * N):
        advance(dt)
        px.append(x[None][0])
        py.append(x[None][1])

    assert px[num_steps // 2 - 1] == approx(-1, abs=2e-2)
    assert py[num_steps // 2 - 1] == approx(0, abs=2e-2)

    assert px[-1] == approx(1, abs=2e-2)
    # TODO: why large error here?
    assert py[-1] == approx(0, abs=3e-2)
