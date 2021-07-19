import taichi as ti


@ti.test(packed=True)
def test_packed_size():
    x = ti.field(ti.i32)
    ti.root.dense(ti.i, 20).dense(ti.ijk, 334).place(x)
    assert x.snode.parent().parent().cell_size_bytes == 149038816
