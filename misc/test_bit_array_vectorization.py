import taichi as ti

ti.init(debug=True, cfg_optimization=False, kernel_profiler=True)

vectorize = False

ci1 = ti.type_factory_.get_custom_int_type(1, False)

x = ti.field(dtype=ci1)
y = ti.field(dtype=ci1)

N = 4096
block_size = 4
bits = 32
boundary_offset = 64

block = ti.root.pointer(ti.ij, (block_size, block_size))
block.dense(ti.ij, (N // block_size, N // (bits * block_size)))._bit_array(
    ti.j, bits, num_bits=bits).place(x)
block.dense(ti.ij, (N // block_size, N // (bits * block_size)))._bit_array(
    ti.j, bits, num_bits=bits).place(y)


@ti.kernel
def init():
    for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                           (boundary_offset, N - boundary_offset)):
        x[i, j] = (N * i + j) % 2


@ti.kernel
def assign_vectorized():
    ti.bit_vectorize(32)
    for i, j in x:
        y[i, j] = x[i, j]


@ti.kernel
def assign_naive():
    for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                           (boundary_offset, N - boundary_offset)):
        y[i, j] = x[i, j]


@ti.kernel
def verify():
    for i, j in ti.ndrange((boundary_offset, N - boundary_offset), (boundary_offset, N - boundary_offset)):
        assert y[i, j] == (N * i + j) % 2


init()
if vectorize:
    assign_vectorized()
else:
    assign_naive()
verify()

ti.kernel_profiler_print()
