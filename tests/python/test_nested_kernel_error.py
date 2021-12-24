import pytest

import taichi as ti


@ti.test()
def test_nested_kernel_error():
    @ti.kernel
    def B():
        pass

    @ti.kernel
    def A():
        B()

    with pytest.raises(ti.TaichiCompilationError):
        A()
