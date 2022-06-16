import numpy as np

import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan)
def test_ndarray_int():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'pos', ti.i32)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n, ))
    g.run({'pos': a})
    assert (a.to_numpy() == np.ones(4)).all()
