import taichi as ti

ti.init(arch=ti.cuda)


@ti.kernel
def p() -> ti.f32:
    print(42)
    return 40 + 2


print(p())
