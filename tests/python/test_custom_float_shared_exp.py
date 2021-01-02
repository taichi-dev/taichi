import taichi as ti
from pytest import approx


# @ti.test(require=ti.extension.quant)
def test_shared_exponents():
    exp = ti.type_factory.custom_int(8, False)
    cit1 = ti.type_factory.custom_int(10, True)
    cit2 = ti.type_factory.custom_int(14, True)
    cft1 = ti.type_factory.custom_float(significand_type=cit1,
                                        exponent_type=exp,
                                        scale=1)
    cft2 = ti.type_factory.custom_float(significand_type=cit2,
                                        exponent_type=exp,
                                        scale=1)
    a = ti.field(dtype=cft1)
    b = ti.field(dtype=cft2)
    ti.root._bit_struct(num_bits=32).place(a, b, shared_exponent=True)

    assert a[None] == 0.0
    assert b[None] == 0.0

    a[None] = 10
    assert a[None] == 10.0
    assert b[None] == 0.0

    a[None] = 0
    assert a[None] == 0.0
    assert b[None] == 0.0

    @ti.kernel
    def foo(x: ti.f32, y: ti.f32):
        a[None] = x
        b[None] = y

    foo(3.2, 0.25)

    assert a[None] == approx(3.2, rel=1e-3)
    assert b[None] == approx(0.25, rel=2e-2)
    a[None] = 0.27
    assert a[None] == approx(0.27, rel=1e-2)
    assert b[None] == approx(0.25, rel=2e-2)
    a[None] = 100
    assert a[None] == approx(100, rel=1e-3)
    assert b[None] == approx(0.25, rel=1e-2)

    b[None] = 0
    assert a[None] == approx(100, rel=1e-3)
    assert b[None] == 0

    foo(0, 0)
    assert a[None] == 0.0
    assert b[None] == 0.0

    # test flush to zero

    foo(1000, 1e-6)
    assert a[None] == 1000.0
    assert b[None] == 0.0

    foo(1000, 1000)
    assert a[None] == 1000.0
    assert b[None] == 1000.0

    foo(1e-30, 1e-30)
    assert a[None] == approx(1e-30, 1e-3)
    assert b[None] == approx(1e-30, 1e-4)


ti.init()
test_shared_exponents()

# TODO: test exp not 8 bits
# TODO: test negative
# TODO: test shared exponent floats with custom int in a single bit struct
