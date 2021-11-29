import taichi as ti


@ti.test()
@ti.must_throw(ti.TaichiError)
def test_nested_kernel_error():
    @ti.kernel
    def B():
        pass

    @ti.kernel
    def A():
        B()

    A()
