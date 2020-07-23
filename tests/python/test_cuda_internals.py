import taichi as ti

# TODO: these are not really tests...


def test_do_nothing():
    ti.init(arch=ti.cuda)

    @ti.kernel
    def test():
        for i in range(10):
            ti.call_internal("do_nothing")

    test()


def test_active_mask():
    ti.init(arch=ti.cuda)

    @ti.kernel
    def test():
        for i in range(48):
            if i % 2 == 0:
                ti.call_internal("test_active_mask")

    test()


def test_shfl_down():
    ti.init(arch=ti.cuda)

    @ti.kernel
    def test():
        for i in range(32):
            ti.call_internal("test_shfl")

    test()
