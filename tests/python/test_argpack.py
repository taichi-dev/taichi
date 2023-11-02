import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_argpack_basic():
    pack_type = ti.types.argpack(a=ti.i32, b=bool, c=ti.f32)
    pack1 = pack_type(a=1, b=False, c=2.1)
    pack2 = pack_type(a=2, b=True, c=2.1)

    @ti.kernel
    def foo(pack: pack_type) -> ti.f32:
        tmp = 0.0
        if pack.b:
            tmp = pack.a + pack.c
        else:
            tmp = pack.a * pack.c
        return tmp

    assert foo(pack1) == test_utils.approx(1 * 2.1, rel=1e-3)
    assert foo(pack2) == test_utils.approx(2 + 2.1, rel=1e-3)


@test_utils.test()
def test_argpack_with_struct():
    struct_type = ti.types.struct(a=ti.i32, c=ti.f32)
    pack_type = ti.types.argpack(d=ti.f32, element=struct_type)

    @ti.kernel
    def foo(pack: pack_type) -> ti.f32:
        tmp = pack.element.a + pack.element.c
        return tmp + pack.d

    pack = pack_type(d=2.1, element=struct_type(a=2, c=2.1))
    assert foo(pack) == test_utils.approx(2 + 2.1 + 2.1, rel=1e-3)


@test_utils.test()
def test_argpack_with_vector():
    pack_type = ti.types.argpack(a=ti.i32, b=ti.types.vector(3, ti.f32), c=ti.f32)
    pack = pack_type(a=1, b=[1.0, 2.0, 3.0], c=2.1)

    @ti.kernel
    def foo(pack: pack_type) -> ti.f32:
        tmp = pack.a * pack.c
        return tmp + pack.b[1]

    assert foo(pack) == test_utils.approx(1 * 2.1 + 2.0, rel=1e-3)


@test_utils.test()
def test_argpack_multiple():
    arr = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    arr.fill([1.0, 2.0, 3.0])

    pack_type1 = ti.types.argpack(a=ti.i32, c=ti.f32)
    pack_type2 = ti.types.argpack(a=ti.types.ndarray(dtype=ti.math.vec3, ndim=2))
    pack1 = pack_type1(a=1, c=2.1)
    pack2 = pack_type2(a=arr)

    @ti.kernel
    def foo(p1: pack_type1, p2: pack_type2) -> ti.f32:
        tmp = p1.a * p1.c
        return tmp + p2.a[1, 2][1]

    assert foo(pack1, pack2) == test_utils.approx(1 * 2.1 + 2.0, rel=1e-3)


@test_utils.test()
def test_argpack_nested():
    arr = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    arr.fill([1.0, 2.0, 3.0])

    pack_type_inner = ti.types.argpack(a=ti.i32, b=ti.i32)
    pack_type = ti.types.argpack(a=ti.types.ndarray(dtype=ti.math.vec3, ndim=2), b=ti.i32, c=pack_type_inner)
    pack_inner = pack_type_inner(a=123, b=456)
    pack = pack_type(a=arr, b=233, c=pack_inner)

    @ti.kernel
    def p(x: pack_type) -> ti.math.vec3:
        return x.a[2, 3]

    @ti.kernel
    def q(x: pack_type) -> int:
        return x.c.a + x.c.b

    @ti.kernel
    def h(x: pack_type) -> int:
        return x.b

    assert p(pack) == [1.0, 2.0, 3.0]
    assert q(pack) == 123 + 456
    assert h(pack) == 233


@test_utils.test()
def test_argpack_as_return():
    pack_type = ti.types.argpack(a=ti.i32, b=bool)

    with pytest.raises(ti.TaichiSyntaxError):

        @ti.kernel
        def foo(pack: pack_type) -> pack_type:
            return pack

        foo()


@test_utils.test()
def test_argpack_as_struct_type_element():
    with pytest.raises(ValueError, match="Invalid data type <ti.ArgPackType a=i32, b=u1>"):
        pack_type = ti.types.argpack(a=ti.i32, b=bool)
        struct_with_argpack_inside = ti.types.struct(element=pack_type)
        print(struct_with_argpack_inside)


@test_utils.test()
def test_argpack_with_ndarray():
    arr = ti.ndarray(dtype=ti.math.vec3, shape=(4, 4))
    arr.fill([1.0, 2.0, 3.0])

    pack_type = ti.types.argpack(element=ti.types.ndarray(dtype=ti.math.vec3, ndim=2))
    pack = pack_type(element=arr)

    @ti.kernel
    def foo(pack: pack_type) -> ti.math.vec3:
        return pack.element[0, 0]

    assert foo(pack) == [1.0, 2.0, 3.0]
