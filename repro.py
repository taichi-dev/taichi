import taichi as ti

ti.init(ti.vulkan, log_level=ti.TRACE)


@ti.kernel
def test() -> ti.i32:
    print("hello")
    return 3


a = test()
print(a)
