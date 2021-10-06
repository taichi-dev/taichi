import taichi as ti


@ti.test(require=ti.extension.packed, packed=True)
def test_packed_size():
    x = ti.field(ti.i32)
    ti.root.dense(ti.i, 17).dense(ti.ijk, 129).place(x)
    assert x.shape == (17 * 129, 129, 129)
    assert x.snode.parent().parent().cell_size_bytes == 4 * 129**3
