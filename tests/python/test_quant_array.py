import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.quant, debug=True)
def test_1D_quant_array():
    qu1 = ti.types.quant.int(1, False)

    x = ti.field(dtype=qu1)

    N = 32

    ti.root.quant_array(ti.i, N, max_num_bits=32).place(x)

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


@test_utils.test(require=ti.extension.quant, debug=True)
def test_1D_quant_array_negative():
    N = 4
    qi7 = ti.types.quant.int(7)
    x = ti.field(dtype=qi7)
    ti.root.quant_array(ti.i, N, max_num_bits=32).place(x)

    @ti.kernel
    def assign():
        for i in range(N):
            assert x[i] == 0
            x[i] = -i
            assert x[i] == -i

    assign()


@test_utils.test(require=ti.extension.quant, debug=True)
def test_1D_quant_array_fixed():
    qfxt = ti.types.quant.fixed(bits=8, max_value=2)

    x = ti.field(dtype=qfxt)

    N = 4

    ti.root.quant_array(ti.i, N, max_num_bits=32).place(x)

    @ti.kernel
    def set_val():
        for i in range(N):
            x[i] = i * 0.5

    @ti.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == i * 0.5

    set_val()
    verify_val()


@test_utils.test(require=ti.extension.quant, debug=True)
def test_2D_quant_array():
    qu1 = ti.types.quant.int(1, False)

    x = ti.field(dtype=qu1)

    M, N = 4, 8

    ti.root.quant_array(ti.ij, (M, N), max_num_bits=32).place(x)

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


@test_utils.test(require=ti.extension.quant, debug=True)
def test_quant_array_struct_for():
    block_size = 16
    N = 64
    cell = ti.root.pointer(ti.i, N // block_size)
    qi7 = ti.types.quant.int(7)

    x = ti.field(dtype=qi7)
    cell.dense(ti.i, block_size // 4).quant_array(ti.i, 4,
                                                  max_num_bits=32).place(x)

    @ti.kernel
    def activate():
        for i in range(N):
            if i // block_size % 2 == 0:
                x[i] = i

    @ti.kernel
    def assign():
        for i in x:
            x[i] -= 1

    @ti.kernel
    def verify():
        for i in range(N):
            if i // block_size % 2 == 0:
                assert x[i] == i - 1
            else:
                assert x[i] == 0

    activate()
    assign()
    verify()
