import taichi as ti
ti.init()


@ti.host_arch_only
def test_func_closure():
    # TODO: remove this after #1344 is merged:
    def static_assert(x):
        assert x

    def my_test():
        a = 32

        @ti.func
        def foo():
            static_assert(a == 32)

        @ti.kernel
        def func():
            static_assert(a == 32)
            foo()

        def dummy():
            func()

        func()
        dummy()
        return dummy, func

    dummy, func = my_test()
    func()
    dummy()
