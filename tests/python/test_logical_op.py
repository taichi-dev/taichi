import taichi as ti
from tests import test_utils


@test_utils.test(debug=True)
def test_logical_and_i32():
    @ti.kernel
    def func(x: ti.i32, y: ti.i32) -> ti.i32:
        return x and y

    assert func(1, 2) == 2
    assert func(2, 1) == 1
    assert func(0, 1) == 0
    assert func(1, 0) == 0


@test_utils.test(debug=True)
def test_logical_or_i32():
    @ti.kernel
    def func(x: ti.i32, y: ti.i32) -> ti.i32:
        return x or y

    assert func(1, 2) == 1
    assert func(2, 1) == 2
    assert func(1, 0) == 1
    assert func(0, 1) == 1


@test_utils.test(debug=True)
def test_logical_vec_i32():
    vec4d = ti.types.vector(4, ti.i32)

    @ti.kernel
    def p() -> vec4d:
        a = ti.Vector([2, 2, 0, 0])
        b = ti.Vector([1, 0, 1, 0])
        z = a or b
        return z

    @ti.kernel
    def q() -> vec4d:
        a = ti.Vector([2, 2, 0, 0])
        b = ti.Vector([1, 0, 1, 0])
        z = a and b
        return z

    x = p()
    y = q()

    assert x[0] == 1
    assert x[1] == 1
    assert x[2] == 1
    assert x[3] == 0
    assert y[0] == 1
    assert y[1] == 0
    assert y[2] == 0
    assert y[3] == 0


# FIXME: bool vectors not supported on spir-v
@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_logical_vec_bool():
    vec4d = ti.types.vector(4, ti.u1)

    @ti.kernel
    def p() -> vec4d:
        a = ti.Vector([True, True, False, False])
        b = ti.Vector([True, False, True, False])
        z = a or b
        return z

    @ti.kernel
    def q() -> vec4d:
        a = ti.Vector([True, True, False, False])
        b = ti.Vector([True, False, True, False])
        z = a and b
        return z

    x = p()
    y = q()

    assert x[0] == True
    assert x[1] == True
    assert x[2] == True
    assert x[3] == False
    assert y[0] == True
    assert y[1] == False
    assert y[2] == False
    assert y[3] == False
