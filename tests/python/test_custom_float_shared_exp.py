import taichi as ti
from pytest import approx


@ti.test(require=ti.extension.quant)
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

    @ti.kernel
    def foo():
        a[None] = 3.2
        b[None] = 0.25

    foo()

    assert a[None] == approx(3.2, rel=1e-3)
    assert b[None] == approx(0.25, rel=2e-2)
    a[None] = 0.27
    assert a[None] == approx(0.27, rel=1e-2)
    assert b[None] == approx(0.25, rel=2e-2)
    a[None] = 100
    assert a[None] == approx(100, rel=1e-3)
    assert b[None] == approx(0.25, rel=1e-2)


# TODO: test negative
# TODO: test zero
# TODO: test shared exponent floats with custom int in a single bit struct
