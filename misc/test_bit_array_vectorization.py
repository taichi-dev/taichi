import taichi as ti

ti.init(debug=True, print_ir=True, cfg_optimization=False)

ci1 = ti.type_factory_.get_custom_int_type(1, False)

x = ti.field(dtype=ci1)
y = ti.field(dtype=ci1)

N = 64
ti.root._bit_array(ti.i, N, num_bits=64).place(x)
ti.root._bit_array(ti.i, N, num_bits=64).place(y)


@ti.kernel
def foo():
    ti.bit_vectorize(64)
    for i in x:
        y[i] = x[i]


foo()
