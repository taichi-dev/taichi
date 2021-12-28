import taichi as myowntaichi


@myowntaichi.test()
def test_module_import():
    @myowntaichi.kernel
    def func():
        for _ in myowntaichi.static(range(8)):
            pass

    func()
