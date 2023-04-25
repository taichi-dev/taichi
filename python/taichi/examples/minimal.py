import taichi as ti

ti.init(arch=ti.vulkan)


@ti.kernel
def p() -> ti.u1:
    print(42)
    return False


print(p())
