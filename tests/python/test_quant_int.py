import taichi as ti
from tests import test_utils


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_int_implicit_cast():
    qi13 = ti.types.quant.int(13, True)
    x = ti.field(dtype=qi13)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    @ti.kernel
    def foo():
        x[None] = 10.3

    foo()
    assert x[None] == 10
