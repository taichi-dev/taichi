import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_nested_kernel_error():
    @ti.kernel
    def B():
        pass

    @ti.kernel
    def A():
        B()

    with pytest.raises(ti.TaichiCompilationError):
        A()
