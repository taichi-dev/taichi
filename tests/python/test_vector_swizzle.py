import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_vector_swizzle_python():
    v = ti.math.vec3(0)
    v = ti.math.vec3(0, 0, 0)
    v = ti.math.vec3([0, 0], 0)
    v = ti.math.vec3(0, v.xx)
    v = ti.math.vec3(0, v.xy)
    v.rgb += 1
    assert all(v.xyz == (1, 1, 1))
    v.zyx += ti.math.vec3(1)
    assert all(v.stp == ti.math.vec3(2, 2, 2))
    assert v.x == 2
    assert v.r == 2
    assert v.s == 2
    w = ti.floor(v)
    assert all(w == v)
    z = ti.math.vec4(w.xyz, 2)
    assert all(z == w.xxxx)


@test_utils.test(debug=True)
def test_vector_swizzle_taichi():
    @ti.kernel
    def foo():
        v = ti.math.vec3(0)
        v = ti.math.vec3(0, 0, 0)
        v = ti.math.vec3([0, 0], 0)
        v = ti.math.vec3(0, v.xx)
        v = ti.math.vec3(0, v.xy)
        v.rgb += 1
        assert all(v.xyz == (1, 1, 1))
        v.zyx += ti.math.vec3(1)
        assert all(v.stp == ti.math.vec3(2, 2, 2))
        assert v.x == 2
        assert v.r == 2
        assert v.s == 2
        w = ti.floor(v).yxz
        assert all(w == v)
        z = ti.math.vec4(w.xyz, 2)
        assert all(z == w.xxxx)

    foo()


@test_utils.test(debug=True)
def test_vector_dtype():
    @ti.kernel
    def foo():
        a = ti.math.vec3(1, 2, 3)
        a /= 2
        assert all(abs(a - (0.5, 1., 1.5)) < 1e-6)
        b = ti.math.ivec3(1.5, 2.5, 3.5)
        assert all(b == (1, 2, 3))

    foo()
