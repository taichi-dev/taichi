import taichi as ti


@ti.test(arch=ti.cpu)
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

    assert n1.cell_size_bytes == 2
    assert 12 <= n2.cell_size_bytes <= 16
    assert n3.cell_size_bytes == 16


@ti.test(arch=ti.cpu)
def test_bit_struct():
    cit = ti.quant.int(16, False)
    x = ti.field(dtype=cit)
    y = ti.field(dtype=ti.type_factory.custom_float(significand_type=cit))
    z = ti.field(dtype=ti.f32)

    n1 = ti.root.dense(ti.i, 32)
    n1.bit_struct(num_bits=32).place(x)

    n2 = ti.root.dense(ti.i, 4)
    n2.bit_struct(num_bits=32).place(y)
    n2.place(z)

    assert n1.cell_size_bytes == 4
    assert n2.cell_size_bytes == 8
