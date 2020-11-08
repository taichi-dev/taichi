import taichi as ti


@ti.test(ti.cpu)
def test_func_closure():
    def my_test():
        a = 32

        @ti.func
        def foo():
            ti.static_assert(a == 32)

        @ti.kernel
        def func():
            ti.static_assert(a == 32)
            foo()

        def dummy():
            func()

        func()
        dummy()
        return dummy, func

    dummy, func = my_test()
    func()
    dummy()
