import taichi as ti

ti.init()

cft1 = ti.quant.float(exp=7, frac=10)
cft2 = ti.quant.float(exp=7, frac=14)
a = ti.field(dtype=cft1)
b = ti.field(dtype=cft2)
ti.root.bit_struct(num_bits=32).place(a, b, shared_exponent=True)

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

print(a[None])

