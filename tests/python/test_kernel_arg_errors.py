import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_pass_float_as_i32():
    @ti.kernel
    def foo(a: ti.i32):
        pass

    with pytest.raises(
            ti.TaichiRuntimeTypeError,
            match=
            r"Argument 0 \(type=<class 'float'>\) cannot be converted into required type i32"
    ) as e:
        foo(1.2)
