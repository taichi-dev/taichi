import pytest
from taichi.lang import impl

import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.cpu])
def test_local_matrix_non_constant_index_real_matrix():
    N = 1
    x = ti.Vector.field(3, float, shape=1)

    @ti.kernel
    def test_invariant_cache():
        for i in range(1):
            x[i][1] = x[i][1] + 1.0
            for j in range(1):
                x[i][1] = x[i][1] - 5.0
                for z in range(1):
                    idx = 0
                    if z == 0:
                        idx = 1
                    x_print = x[i][idx]

                    assert x_print == x[i][1]

    test_invariant_cache()
