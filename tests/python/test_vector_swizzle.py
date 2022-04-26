import pytest
import re

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
def test_vector_swizzle2_taichi():
    @ti.kernel
    def foo():
        v = ti.math.vec3(0, 0, 0)
        v.brg += 1
        assert all(v.xyz == (1, 1, 1))
        v.x = 1
        v.g = 2
        v.p = 3
        v123 = ti.math.vec3(1, 2, 3)
        v231 = ti.math.vec3(2, 3, 1)
        v113 = ti.math.vec3(1, 1, 3)
        assert all(v == v123)
        assert all(v.xyz == v123)
        assert all(v.rgb == v123)
        assert all(v.stp == v123)
        assert all(v.yzx == v231)
        assert all(v.gbr == v231)
        assert all(v.tps == v231)
        assert all(v.xxz == v113)
        assert all(v.rrb == v113)
        assert all(v.ssp == v113)
        v.bgr = v123
        v321 = ti.math.vec3(3, 2, 1)
        assert all(v.xyz == v321)
        assert all(v.rgb == v321)
        assert all(v.stp == v321)

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

@test_utils.test()
def test_vector_invalid_swizzle_patterns():
    a = ti.math.vec2(1, 2)
    with pytest.raises(
            ti.TaichiSyntaxError,
            match=re.escape("vec2 only has attributes=('x', 'y'), got=('z',)")):
        a.z = 3
    with pytest.raises(
            ti.TaichiSyntaxError,
            match=re.escape("vec2 only has attributes=('x', 'y'), got=('x', 'y', 'z')")):
        a.xyz = [1, 2, 3]

    with pytest.raises(
            ti.TaichiCompilationError,
            match=re.escape("value len does not match the swizzle pattern=xy")):
        a.xy = [1, 2, 3]