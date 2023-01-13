import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_packed_size():
    x = ti.field(ti.i32)
    ti.root.dense(ti.l, 3).dense(ti.ijk, 129).place(x)
    assert x.shape == (129, 129, 129, 3)
    assert x.snode.parent().parent()._cell_size_bytes == 4 * 129**3
