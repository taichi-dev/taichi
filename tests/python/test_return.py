import taichi as ti


@ti.must_throw(ti.TaichiSyntaxError)
def test_return_in_kernel():
    @ti.kernel
    def kernel():
        return 1

    kernel()
