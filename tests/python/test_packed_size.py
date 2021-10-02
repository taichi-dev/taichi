import taichi as ti


@ti.test(require=ti.extension.packed, packed=True)
def test_packed_size():
    x = ti.field(ti.i32)
    ti.root.dense(ti.i, 17).dense(ti.ijk, 73).place(x)
    assert x.shape == (17 * 73, 73, 73)
    # Where does this 4 come from?
    assert x.snode.parent().parent().cell_size_bytes == 4 * 73**3
