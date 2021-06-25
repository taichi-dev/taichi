import pytest

import taichi as ti


@ti.test(arch=ti.cpu)
def test_pass_float_as_i32():
    @ti.kernel
    def foo(a: ti.i32):
        pass

    with pytest.raises(ti.KernelArgError) as e:
        foo(1.2)

    assert e.type is ti.KernelArgError
    assert e.value.args[
        0] == "Argument 0 (type=<class 'float'>) cannot be converted into required type i32"
