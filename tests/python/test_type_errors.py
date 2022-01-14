import numpy as np
import pytest

import taichi as ti


@ti.test()
def test_0d_ndarray():
    @ti.kernel
    def foo() -> ti.i32:
        a = np.array(3, dtype=np.int32)
        return a

    assert foo() == 3


@ti.test()
def test_non_0d_ndarray():
    @ti.kernel
    def foo():
        a = np.array([1])

    with pytest.raises(
            ti.TaichiTypeError,
            match=
            "Only 0-dimensional numpy array can be used to initialize a scalar expression"
    ):
        foo()
