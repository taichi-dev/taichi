import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_int_implicit_cast():
    qi13 = ti.types.quant.int(13, True)
    x = ti.field(dtype=qi13)

    ti.root.bit_struct(num_bits=32).place(x)

    @ti.kernel
    def foo():
        x[None] = 10.3

    foo()
    assert x[None] == 10
