import taichi as ti
from tests import test_utils


@test_utils.test()
def test_empty():
    @ti.kernel
    def func():
        pass

    func()


@test_utils.test()
def test_empty_args():
    @ti.kernel
    def func(x: ti.i32, arr: ti.ext_arr()):
        pass

    import numpy as np
    func(42, np.arange(10, dtype=np.float32))
