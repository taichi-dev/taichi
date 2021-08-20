import pytest

import taichi as ti


@pytest.mark.parametrize('x,y', [(2, 3), (-1, 4)])
@ti.test(exclude=ti.cpu, require=ti.extension.extfunc)
def test_asm(x, y):
    @ti.kernel
    def func(x: ti.f32, y: ti.f32) -> ti.f32:
        z = 0.0
        ti.asm('$0 = %0 * %1', inputs=[x, y], outputs=[z])
        return z

    assert func(x, y) == x * y
