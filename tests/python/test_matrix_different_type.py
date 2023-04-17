import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.quant_basic)
def test_valid():
    qflt = ti.types.quant.float(exp=8, frac=5, signed=True)
    qfxt = ti.types.quant.fixed(bits=10, signed=True, scale=0.1)
    type_list = [[qflt, qfxt], [qflt, qfxt]]
    a = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    b = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    c = ti.Matrix.field(len(type_list), len(type_list[0]), dtype=type_list)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(a.get_scalar_field(0, 0), a.get_scalar_field(0, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(a.get_scalar_field(1, 0), a.get_scalar_field(1, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(b.get_scalar_field(0, 0), b.get_scalar_field(0, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(b.get_scalar_field(1, 0), b.get_scalar_field(1, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(c.get_scalar_field(0, 0), c.get_scalar_field(0, 1))
    ti.root.dense(ti.i, 1).place(bitpack)
    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(c.get_scalar_field(1, 0), c.get_scalar_field(1, 1))
    ti.root.dense(ti.i, 1).place(bitpack)

    @ti.kernel
    def init():
        a[0] = [[1.0, 3.0], [2.0, 1.0]]
        b[0] = [[2.0, 4.0], [-2.0, 1.0]]
        c[0] = a[0] + b[0]

    def verify():
        assert c[0][0, 0] == pytest.approx(3.0)
        assert c[0][0, 1] == pytest.approx(7.0)
        assert c[0][1, 0] == pytest.approx(0.0)
        assert c[0][1, 1] == pytest.approx(2.0)

    init()
    verify()


@test_utils.test(require=ti.extension.quant_basic)
def test_invalid():
    qit = ti.types.quant.int(bits=10, signed=True)
    qfxt = ti.types.quant.fixed(bits=10, signed=True, scale=0.1)
    type_list = [qit, qfxt]
    with pytest.raises(
        RuntimeError,
        match="Member fields of a matrix field must have the same compute type",
    ):
        a = ti.Vector.field(len(type_list), dtype=type_list)
