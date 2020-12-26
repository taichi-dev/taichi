import taichi as ti
import numpy as np


@ti.test(require=ti.extension.quant, debug=True, cfg_optimization=False)
def test_vectorized_struct_for():
    ci1 = ti.type_factory_.get_custom_int_type(1, False)

    x = ti.field(dtype=ci1)
    y = ti.field(dtype=ci1)

    N = 4096
    n_blocks = 4
    bits = 32
    boundary_offset = 1024

    block = ti.root.pointer(ti.ij, (n_blocks, n_blocks))
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks)))._bit_array(
        ti.j, bits, num_bits=bits).place(x)
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks)))._bit_array(
        ti.j, bits, num_bits=bits).place(y)

    @ti.kernel
    def init():
        for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                               (boundary_offset, N - boundary_offset)):
            x[i, j] = ti.random(dtype=ti.i32) % 2

    @ti.kernel
    def assign_vectorized():
        ti.bit_vectorize(32)
        for i, j in x:
            y[i, j] = x[i, j]

    @ti.kernel
    def verify():
        for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                               (boundary_offset, N - boundary_offset)):
            assert y[i, j] == x[i, j]

    init()
    assign_vectorized()
    verify()
