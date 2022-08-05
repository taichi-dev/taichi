import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.quant_basic, debug=True)
def test_simple_array():
    qi13 = ti.types.quant.int(13, True)
    qu19 = ti.types.quant.int(19, False)

    x = ti.field(dtype=qi13)
    y = ti.field(dtype=qu19)

    N = 12

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x, y)
    ti.root.dense(ti.i, N).place(bitpack)

    @ti.kernel
    def set_val():
        for i in range(N):
            x[i] = -2**i
            y[i] = 2**i - 1

    @ti.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == -2**i
            assert y[i] == 2**i - 1

    set_val()
    verify_val()

    # Test read and write in Python-scope by calling the wrapped, untranslated function body
    set_val.__wrapped__()
    verify_val.__wrapped__()


# TODO: remove excluding of ti.metal
@test_utils.test(require=ti.extension.quant_basic,
                 exclude=[ti.metal],
                 debug=True)
def test_quant_int_load_and_store():
    qi13 = ti.types.quant.int(13, True)
    qu14 = ti.types.quant.int(14, False)
    qi5 = ti.types.quant.int(5, True)

    x = ti.field(dtype=qi13)
    y = ti.field(dtype=qu14)
    z = ti.field(dtype=qi5)

    test_case_np = np.array(
        [[2**12 - 1, 2**14 - 1, -(2**3)], [2**11 - 1, 2**13 - 1, -(2**2)],
         [0, 0, 0], [123, 4567, 8], [10, 31, 11]],
        dtype=np.int32)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x, y, z)
    ti.root.place(bitpack)
    test_case = ti.Vector.field(3, dtype=ti.i32, shape=len(test_case_np))
    test_case.from_numpy(test_case_np)

    @ti.kernel
    def set_val(idx: ti.i32):
        x[None] = test_case[idx][0]
        y[None] = test_case[idx][1]
        z[None] = test_case[idx][2]

    @ti.kernel
    def verify_val(idx: ti.i32):
        assert x[None] == test_case[idx][0]
        assert y[None] == test_case[idx][1]
        assert z[None] == test_case[idx][2]

    for idx in range(len(test_case_np)):
        set_val(idx)
        verify_val(idx)

    # Test read and write in Python-scope by calling the wrapped, untranslated function body
    for idx in range(len(test_case_np)):
        set_val.__wrapped__(idx)
        verify_val.__wrapped__(idx)


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_int_full_struct():
    qit = ti.types.quant.int(32, True)
    x = ti.field(dtype=qit)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.dense(ti.i, 1).place(bitpack)

    x[0] = 15
    assert x[0] == 15

    x[0] = 12
    assert x[0] == 12


def test_bitpacked_fields():
    def test_single_bitpacked_fields(physical_type, compute_type, quant_bits,
                                     test_case):
        ti.init(arch=ti.cpu, debug=True)

        qit1 = ti.types.quant.int(quant_bits[0], True, compute_type)
        qit2 = ti.types.quant.int(quant_bits[1], False, compute_type)
        qit3 = ti.types.quant.int(quant_bits[2], True, compute_type)

        a = ti.field(dtype=qit1)
        b = ti.field(dtype=qit2)
        c = ti.field(dtype=qit3)
        bitpack = ti.BitpackedFields(max_num_bits=physical_type)
        bitpack.place(a, b, c)
        ti.root.place(bitpack)

        @ti.kernel
        def set_val(test_val: ti.types.ndarray()):
            a[None] = test_val[0]
            b[None] = test_val[1]
            c[None] = test_val[2]

        @ti.kernel
        def verify_val(test_val: ti.types.ndarray()):
            assert a[None] == test_val[0]
            assert b[None] == test_val[1]
            assert c[None] == test_val[2]

        set_val(test_case)
        verify_val(test_case)

        ti.reset()

    test_single_bitpacked_fields(8, ti.i8, [3, 3, 2],
                                 np.array([2**2 - 1, 2**3 - 1, -2**1]))
    test_single_bitpacked_fields(16, ti.i16, [4, 7, 5],
                                 np.array([2**3 - 1, 2**7 - 1, -2**4]))
    test_single_bitpacked_fields(32, ti.i32, [17, 11, 4],
                                 np.array([2**16 - 1, 2**10 - 1, -2**3]))
    test_single_bitpacked_fields(64, ti.i64, [32, 23, 9],
                                 np.array([2**31 - 1, 2**23 - 1, -2**8]))
    test_single_bitpacked_fields(32, ti.i16, [7, 12, 13],
                                 np.array([2**6 - 1, 2**12 - 1, -2**12]))
    test_single_bitpacked_fields(64, ti.i32, [18, 22, 24],
                                 np.array([2**17 - 1, 2**22 - 1, -2**23]))

    test_single_bitpacked_fields(16, ti.i16, [5, 5, 6], np.array([15, 5, 20]))
    test_single_bitpacked_fields(32, ti.i32, [10, 10, 12],
                                 np.array([11, 19, 2020]))


@test_utils.test(require=[ti.extension.quant_basic, ti.extension.sparse],
                 debug=True)
def test_bitpacked_fields_struct_for():
    block_size = 16
    N = 64
    cell = ti.root.pointer(ti.i, N // block_size)
    fixed32 = ti.types.quant.fixed(bits=32, max_value=1024)

    x = ti.field(dtype=fixed32)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    cell.dense(ti.i, block_size).place(bitpack)

    for i in range(N):
        if i // block_size % 2 == 0:
            x[i] = 0

    @ti.kernel
    def assign():
        for i in x:
            x[i] = ti.cast(i, float)

    assign()

    for i in range(N):
        if i // block_size % 2 == 0:
            assert x[i] == pytest.approx(i, abs=1e-3)
        else:
            assert x[i] == 0


@test_utils.test(require=ti.extension.quant_basic, debug=True)
def test_multiple_types():
    f15 = ti.types.quant.float(exp=5, frac=10)
    f18 = ti.types.quant.float(exp=5, frac=13)
    u4 = ti.types.quant.int(bits=4, signed=False)

    p = ti.field(dtype=f15)
    q = ti.field(dtype=f18)
    r = ti.field(dtype=u4)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(p, q, shared_exponent=True)
    bitpack.place(r)
    ti.root.dense(ti.i, 12).place(bitpack)

    @ti.kernel
    def set_val():
        for i in p:
            p[i] = i * 3
            q[i] = i * 2
            r[i] = i

    @ti.kernel
    def verify_val():
        for i in p:
            assert p[i] == i * 3
            assert q[i] == i * 2
            assert r[i] == i

    set_val()
    verify_val()


@test_utils.test()
def test_invalid_place():
    f15 = ti.types.quant.float(exp=5, frac=10)
    p = ti.field(dtype=f15)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    with pytest.raises(
            ti.TaichiCompilationError,
            match=
            'At least 2 fields need to be placed when shared_exponent=True'):
        bitpack.place(p, shared_exponent=True)
