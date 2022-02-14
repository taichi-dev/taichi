import taichi as ti
from tests import test_utils


@test_utils.test()
def test_argument_error():
    x = ti.field(ti.i32)

    ti.root.place(x)

    try:

        @ti.kernel
        def set_i32_notype(v):
            pass
    except ti.TaichiSyntaxError:
        pass

    try:

        @ti.kernel
        def set_i32_args(*args):
            pass
    except ti.TaichiSyntaxError:
        pass

    try:

        @ti.kernel
        def set_i32_kwargs(**kwargs):
            pass
    except ti.TaichiSyntaxError:
        pass

    @ti.kernel
    def set_i32(v: ti.i32):
        x[None] = v

    set_i32(123)
    assert x[None] == 123
