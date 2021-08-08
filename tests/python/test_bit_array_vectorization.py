import numpy as np

import taichi as ti


@ti.test(require=ti.extension.quant, debug=True, cfg_optimization=False)
def test_vectorized_struct_for():
    cu1 = ti.quant.int(1, False)

    x = ti.field(dtype=cu1)
    y = ti.field(dtype=cu1)

    N = 4096
    n_blocks = 4
    bits = 32
    boundary_offset = 1024

    block = ti.root.pointer(ti.ij, (n_blocks, n_blocks))
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).bit_array(
        ti.j, bits, num_bits=bits).place(x)
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).bit_array(
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


@ti.test(require=ti.extension.quant)
def test_offset_load():
    ci1 = ti.quant.int(1, False)

    x = ti.field(dtype=ci1)
    y = ti.field(dtype=ci1)
    z = ti.field(dtype=ci1)

    N = 4096
    n_blocks = 4
    bits = 32
    boundary_offset = 1024
    assert boundary_offset >= N // n_blocks

    block = ti.root.pointer(ti.ij, (n_blocks, n_blocks))
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).bit_array(
        ti.j, bits, num_bits=bits).place(x)
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).bit_array(
        ti.j, bits, num_bits=bits).place(y)
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).bit_array(
        ti.j, bits, num_bits=bits).place(z)

    @ti.kernel
    def init():
        for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                               (boundary_offset, N - boundary_offset)):
            x[i, j] = ti.random(dtype=ti.i32) % 2

    @ti.kernel
    def assign_vectorized(dx: ti.template(), dy: ti.template()):
        ti.bit_vectorize(32)
        for i, j in x:
            y[i, j] = x[i + dx, j + dy]
            z[i, j] = x[i + dx, j + dy]

    @ti.kernel
    def verify(dx: ti.template(), dy: ti.template()):
        for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                               (boundary_offset, N - boundary_offset)):
            assert y[i, j] == x[i + dx, j + dy]

    init()
    assign_vectorized(0, 1)
    verify(0, 1)
    assign_vectorized(1, 0)
    verify(1, 0)
    assign_vectorized(0, -1)
    verify(0, -1)
    assign_vectorized(-1, 0)
    verify(-1, 0)
    assign_vectorized(1, 1)
    verify(1, 1)
    assign_vectorized(1, -1)
    verify(1, -1)
    assign_vectorized(-1, -1)
    verify(-1, -1)
    assign_vectorized(-1, 1)
    verify(-1, 1)


@ti.test(require=ti.extension.quant, debug=True)
def test_evolve():
    ci1 = ti.quant.int(1, False)

    x = ti.field(dtype=ci1)
    y = ti.field(dtype=ci1)
    z = ti.field(dtype=ci1)

    N = 4096
    n_blocks = 4
    bits = 32
    boundary_offset = 1024
    assert boundary_offset >= N // n_blocks

    block = ti.root.pointer(ti.ij, (n_blocks, n_blocks))
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).bit_array(
        ti.j, bits, num_bits=bits).place(x)
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).bit_array(
        ti.j, bits, num_bits=bits).place(y)
    block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).bit_array(
        ti.j, bits, num_bits=bits).place(z)

    @ti.kernel
    def init():
        for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                               (boundary_offset, N - boundary_offset)):
            x[i, j] = ti.random(dtype=ti.i32) % 2

    @ti.kernel
    def evolve_vectorized(x: ti.template(), y: ti.template()):
        ti.bit_vectorize(32)
        for i, j in x:
            num_active_neighbors = 0
            num_active_neighbors += ti.cast(x[i - 1, j - 1], ti.u32)
            num_active_neighbors += ti.cast(x[i - 1, j], ti.u32)
            num_active_neighbors += ti.cast(x[i - 1, j + 1], ti.u32)
            num_active_neighbors += ti.cast(x[i, j - 1], ti.u32)
            num_active_neighbors += ti.cast(x[i, j + 1], ti.u32)
            num_active_neighbors += ti.cast(x[i + 1, j - 1], ti.u32)
            num_active_neighbors += ti.cast(x[i + 1, j], ti.u32)
            num_active_neighbors += ti.cast(x[i + 1, j + 1], ti.u32)
            y[i, j] = (num_active_neighbors == 3) or (num_active_neighbors == 2
                                                      and x[i, j] == 1)

    @ti.kernel
    def evolve_naive(x: ti.template(), y: ti.template()):
        for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                               (boundary_offset, N - boundary_offset)):
            num_active_neighbors = 0
            num_active_neighbors += ti.cast(x[i - 1, j - 1], ti.u32)
            num_active_neighbors += ti.cast(x[i - 1, j], ti.u32)
            num_active_neighbors += ti.cast(x[i - 1, j + 1], ti.u32)
            num_active_neighbors += ti.cast(x[i, j - 1], ti.u32)
            num_active_neighbors += ti.cast(x[i, j + 1], ti.u32)
            num_active_neighbors += ti.cast(x[i + 1, j - 1], ti.u32)
            num_active_neighbors += ti.cast(x[i + 1, j], ti.u32)
            num_active_neighbors += ti.cast(x[i + 1, j + 1], ti.u32)
            y[i, j] = (num_active_neighbors == 3) or (num_active_neighbors == 2
                                                      and x[i, j] == 1)

    @ti.kernel
    def verify():
        for i, j in ti.ndrange((boundary_offset, N - boundary_offset),
                               (boundary_offset, N - boundary_offset)):
            assert y[i, j] == z[i, j]

    init()
    evolve_naive(x, z)
    evolve_vectorized(x, y)
    verify()
