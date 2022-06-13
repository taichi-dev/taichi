import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_primitives():
    x = ti.field(dtype=ti.i16)
    y = ti.field(dtype=ti.f32)
    z = ti.field(dtype=ti.f64)

    p = ti.field(dtype=ti.f32)
    q = ti.field(dtype=ti.f32)
    r = ti.field(dtype=ti.f64)

    n1 = ti.root.dense(ti.i, 32)
    n1.place(x)

    n2 = ti.root.dense(ti.i, 32)
    n2.place(y, z)

    n3 = ti.root.dense(ti.i, 1)
    n3.place(p, q, r)

    assert n1._cell_size_bytes == 2
    assert n2._cell_size_bytes in [12, 16]
    assert n3._cell_size_bytes == 16

    assert n1._offset_bytes_in_parent_cell == 0
    assert n2._offset_bytes_in_parent_cell == 2 * 32
    assert n3._offset_bytes_in_parent_cell in [
        2 * 32 + 12 * 32, 2 * 32 + 16 * 32
    ]

    assert x.snode._offset_bytes_in_parent_cell == 0
    assert y.snode._offset_bytes_in_parent_cell == 0
    assert z.snode._offset_bytes_in_parent_cell in [4, 8]
    assert p.snode._offset_bytes_in_parent_cell == 0
    assert q.snode._offset_bytes_in_parent_cell == 4
    assert r.snode._offset_bytes_in_parent_cell == 8


@test_utils.test(arch=ti.cpu)
def test_bit_struct():
    x = ti.field(dtype=ti.types.quant.int(16, False))
    y = ti.field(dtype=ti.types.quant.fixed(16, False))
    z = ti.field(dtype=ti.f32)

    n1 = ti.root.dense(ti.i, 32)
    n1.bit_struct(num_bits=32).place(x)

    n2 = ti.root.dense(ti.i, 4)
    n2.bit_struct(num_bits=32).place(y)
    n2.place(z)

    assert n1._cell_size_bytes == 4
    assert n2._cell_size_bytes == 8
