import taichi as ti
import math
# from pytest import approx


# @ti.test(require=ti.extension.quant)
# def test_custom_float():

ti.init()
ci13 = ti.type_factory_.get_custom_int_type(13, True)
ci8 = ti.type_factory_.get_custom_int_type(8, True)
cft = ti.type_factory.custom_float(significand_type=ci13, compute_type=ti.f32.get_ptr(), scale=0.1)
x = ti.field(dtype=cft)
y = ti.field(dtype=cft)
z = ti.field(dtype=cft)

ti.root._bit_struct(num_bits=32).place(x, y, z, shared_exponent=True)

@ti.kernel
def foo():
    x[None] = 0.7
    print(x[None])
    x[None] = x[None] + 0.4

foo()
assert x[None] == approx(1.1)
x[None] = 0.64
assert x[None] == approx(0.6)
x[None] = 0.66
assert x[None] == approx(0.7)
