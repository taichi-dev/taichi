import taichi as ti

# ti.init(debug=True, cfg_optimization=False)
ti.init(debug=True, cfg_optimization=False, print_ir=True)

ci1 = ti.type_factory_.get_custom_int_type(1, False)

x = ti.field(dtype=ci1)
y = ti.field(dtype=ci1)

N = 4096
block_size = 4
bits = 32

ti.root.pointer(ti.ij, (block_size, block_size)).dense(ti.ij, (N // block_size, N // (bits * block_size)))._bit_array(ti.j, bits,
                                                num_bits=bits).place(x)
ti.root.pointer(ti.ij, (block_size, block_size)).dense(ti.ij, (N // block_size, N // (bits * block_size)))._bit_array(ti.j, bits,
                                                num_bits=bits).place(y)


# @ti.kernel
# def init():
#     for i, j in ti.ndrange(N, N):
#         x[i, j] = (N * i + j) % 2


@ti.kernel
def assign():
    for i, j in x:
        y[i, j] = x[i, j]


# @ti.kernel
# def verify():
#     for i, j in ti.ndrange(N, N):
#         assert y[i, j] == (N * i + j) % 2


# init()
assign()
# verify()
