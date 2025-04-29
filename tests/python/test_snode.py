import taichi as ti
from tests import test_utils


@ti.kernel
def some_kernel(_: ti.template()): ...


@test_utils.test(cpu_max_num_threads=1)
def test_get_snode_tree_id():
    s = ti.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 0

    s = ti.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 1

    s = ti.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 2
