import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_ifexpr():
    n_grids = 10

    g_v = ti.Vector.field(3, float, (n_grids, n_grids, n_grids))
    g_m = ti.field(float, (n_grids, n_grids, n_grids))

    @ti.kernel
    def func():
        for I in ti.grouped(g_m):
            cond = (I < 3) & (g_v[I] < 0) | (I > n_grids - 3) & (g_v[I] > 0)
            g_v[I] = 0 if cond else g_v[I]

    with pytest.raises(ti.TaichiSyntaxError,
                       match='Please use "ti.select" instead.'):
        func()
