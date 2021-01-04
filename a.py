import taichi as ti

ti.init(arch=ti.gpu)

exp = ti.type_factory.custom_int(5, False)
cit = ti.type_factory.custom_int(22, True)
cft = ti.type_factory.custom_float(significand_type=cit, exponent_type=exp, scale=1)
v = ti.field(dtype=cft)
ti.root._bit_struct(num_bits=32).place(v)

@ti.kernel
def set_val():
    v[None] = 1.0

@ti.kernel
def p():
    print(v[None])

set_val()
p()
