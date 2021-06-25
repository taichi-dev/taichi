import pytest

import taichi as ti


@ti.require(ti.extension.extfunc)
@ti.archs_excluding(ti.cpu)
@pytest.mark.parametrize('x,y', [(2, 3), (-1, 4)])
def test_asm(x, y):
    @ti.kernel
    def func(x: ti.f32, y: ti.f32) -> ti.f32:
        z = 0.0
        ti.asm('$0 = %0 * %1', inputs=[x, y], outputs=[z])
        return z

    assert func(x, y) == x * y
