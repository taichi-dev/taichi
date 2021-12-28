import taichi as ti


@ti.test(debug=True, short_circuit_operators=True)
def test_and_shorted():
    a = ti.field(ti.i32, shape=10)

    @ti.func
    def explode() -> ti.i32:
        return a[-1]

    @ti.kernel
    def func() -> ti.i32:
        return False and explode()

    assert func() == 0


@ti.test(debug=True, short_circuit_operators=True)
def test_and_not_shorted():

    @ti.kernel
    def func() -> ti.i32:
        return True and False

    assert func() == 0


@ti.test(debug=True, short_circuit_operators=True)
def test_or_shorted():
    a = ti.field(ti.i32, shape=10)

    @ti.func
    def explode() -> ti.i32:
        return a[-1]

    @ti.kernel
    def func() -> ti.i32:
        return True or explode()

    assert func() == 1


@ti.test(debug=True, short_circuit_operators=True)
def test_or_not_shorted():

    @ti.kernel
    def func() -> ti.i32:
        return False or True

    assert func() == 1
