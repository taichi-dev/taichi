import taichi as ti
from tests import test_utils


@test_utils.test(debug=True)
def test_bool_type_anno():
    @ti.func
    def f(x: bool) -> bool:
        return not x

    @ti.kernel
    def test():
        assert f(True) == False
        assert f(False) == True

    test()


@test_utils.test(debug=True)
def test_bool_type_conv():
    @ti.func
    def f(x: ti.u32) -> bool:
        return bool(x)

    @ti.kernel
    def test():
        assert f(1000) == True
        assert f(ti.u32(4_294_967_295)) == True

    test()
