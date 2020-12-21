import taichi as ti

ti.init(debug=True, cfg_optimization=False)
# ti.init(debug=True, cfg_optimization=False, print_ir=True)

ci1 = ti.type_factory_.get_custom_int_type(1, False)

x = ti.field(dtype=ci1)
y = ti.field(dtype=ci1)

N = 1024
bits = 32
ti.root.dense(ti.ij,  (N, N // bits))._bit_array(ti.j, bits, num_bits=bits).place(x)
ti.root.dense(ti.ij,  (N, N // bits))._bit_array(ti.j, bits, num_bits=bits).place(y)


@ti.kernel
def init():
    for i, j in ti.ndrange(N, N):
        x[i, j] = (N * i + j) % 2


@ti.kernel
def assign():
    for i in ti.grouped(x):
        y[i] = x[i]


@ti.kernel
def verify():
    for i, j in ti.ndrange(N, N):
        assert y[i, j] == (N * i + j) % 2


init()
assign()
verify()
