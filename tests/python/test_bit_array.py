import numpy as np

import taichi as ti


@ti.test(require=ti.extension.quant, debug=True)
def test_1D_bit_array():
    cu1 = ti.quant.int(1, False)

    x = ti.field(dtype=cu1)

    N = 32

    ti.root.bit_array(ti.i, N, num_bits=32).place(x)

    ti.get_runtime().materialize()

    @ti.kernel
    def set_val():
        for i in range(N):
            x[i] = i % 2

    @ti.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == i % 2

    set_val()
    verify_val()


@ti.test(require=ti.extension.quant, debug=True)
def test_2D_bit_array():
    ci1 = ti.quant.int(1, False)

    x = ti.field(dtype=ci1)

    M, N = 4, 8

    ti.root.bit_array(ti.ij, (M, N), num_bits=32).place(x)

    ti.get_runtime().materialize()

    @ti.kernel
    def set_val():
        for i in range(M):
            for j in range(N):
                x[i, j] = (i * N + j) % 2

    @ti.kernel
    def verify_val():
        for i in range(M):
            for j in range(N):
                assert x[i, j] == (i * N + j) % 2

    set_val()
    verify_val()
