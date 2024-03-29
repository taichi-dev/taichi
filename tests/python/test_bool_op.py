import taichi as ti
from tests import test_utils


@test_utils.test(debug=True, short_circuit_operators=True)
def test_and_shorted():
    a = ti.field(ti.i32, shape=10)

    @ti.func
    def explode() -> ti.u1:
        return ti.u1(a[-1])

    @ti.kernel
    def func() -> ti.u1:
        return False and explode()

    assert func() == False


@test_utils.test(debug=True, short_circuit_operators=True)
def test_and_not_shorted():
    @ti.kernel
    def func() -> ti.i32:
        return True and False

    assert func() == 0


@test_utils.test(debug=True, short_circuit_operators=True)
def test_or_shorted():
    a = ti.field(ti.i32, shape=10)

    @ti.func
    def explode() -> ti.u1:
        return ti.u1(a[-1])

    @ti.kernel
    def func() -> ti.i32:
        return True or explode()

    assert func() == 1


@test_utils.test(debug=True, short_circuit_operators=True)
def test_or_not_shorted():
    @ti.kernel
    def func() -> ti.u1:
        return False or True

    assert func() == 1


@test_utils.test(debug=True)
def test_static_or():
    @ti.kernel
    def func() -> ti.i32:
        return ti.static(0 or 3 or 5)

    assert func() == 3


@test_utils.test(debug=True)
def test_static_and():
    @ti.kernel
    def func() -> ti.i32:
        return ti.static(5 and 2 and 0)

    assert func() == 0


@test_utils.test(require=ti.extension.data64, default_ip=ti.i64)
def test_condition_type():
    @ti.kernel
    def func() -> int:
        x = False
        result = 0
        if x:
            result = 1
        else:
            result = 2
        return result

    assert func() == 2


@test_utils.test(require=ti.extension.data64, default_ip=ti.i64)
def test_u1_bool():
    @ti.kernel
    def func() -> ti.u1:
        return True

    assert func() == 1


@test_utils.test()
def test_bool_parameter():
    @ti.kernel
    def func(x: ti.u1) -> ti.u1:
        return not x

    assert func(False) == True


@test_utils.test()
def test_if():
    @ti.kernel
    def func(x: ti.u1) -> ti.u1:
        y = False
        if x:
            y = True
        return y

    assert func(2 == 2) == True
    assert func(2 == 3) == False


@test_utils.test()
def test_ternary():
    @ti.kernel
    def func(x: ti.i32) -> ti.u1:
        return True if x == 114514 else False

    assert func(114514) == True
    assert func(1919810) == False
