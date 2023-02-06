import taichi as ti
from tests import test_utils


@test_utils.test()
def test_cfg_continue():
    x = ti.field(dtype=int, shape=1)
    state = ti.field(dtype=int, shape=1)

    @ti.kernel
    def foo():
        for p in range(1):
            if state[p] == 0:
                x[p] = 1
                continue

            x[p] = 100

    foo()
    assert x[0] == 1
