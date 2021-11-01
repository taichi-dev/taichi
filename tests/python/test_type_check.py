import pytest
import taichi as ti


@ti.test(arch=ti.cpu)
def test_():
    @ti.kernel
    def bitwise_float():
        a = 1
        b = 3.1
        c = a & b

    with pytest.raises(SystemExit):
        bitwise_float()
