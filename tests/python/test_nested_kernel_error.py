import taichi as ti


@ti.must_throw(ti.TaichiSyntaxError)
def test_nested_kernel_error():
    @ti.kernel
    def B():
        pass

    @ti.kernel
    def A():
        B()

    A()
