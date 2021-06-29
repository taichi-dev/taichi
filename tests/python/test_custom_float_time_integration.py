import math

import pytest
from pytest import approx

import taichi as ti


@pytest.mark.parametrize('use_cft,use_exponent,use_shared_exp',
                         [(False, False, False), (True, False, False),
                          (True, True, False), (True, True, True)])
@ti.test(require=ti.extension.quant)
def test_custom_float_time_integration(use_cft, use_exponent, use_shared_exp):
    if use_cft:
        if use_exponent:
            exp = ti.quant.int(6, False)
            cit = ti.quant.int(13, True)
            cft = ti.type_factory.custom_float(significand_type=cit,
                                               exponent_type=exp,
                                               scale=1)
            x = ti.Vector.field(2, dtype=cft)
            if use_shared_exp:
                ti.root.bit_struct(num_bits=32).place(x, shared_exponent=True)
            else:
                ti.root.bit_struct(num_bits=32).place(x(0))
                ti.root.bit_struct(num_bits=32).place(x(1))
        else:
            cit = ti.quant.int(16, True)
            cft = ti.type_factory.custom_float(significand_type=cit,
                                               scale=1 / 2**14)
            x = ti.Vector.field(2, dtype=cft)
            ti.root.bit_struct(num_bits=32).place(x)
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
