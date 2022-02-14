import taichi as myowntaichi
from tests import test_utils


@test_utils.test()
def test_module_import():
    @myowntaichi.kernel
    def func():
        for _ in myowntaichi.static(range(8)):
            pass

    func()
