import taichi as ti


@ti.all_archs
def test_empty():
    @ti.kernel
    def func():
        pass

    func()


@ti.all_archs
def test_empty_args():
    @ti.kernel
    def func(x: ti.i32, arr: ti.ext_arr()):
        pass

    import numpy as np
    func(42, np.arange(10, dtype=np.float32))
