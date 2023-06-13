import taichi as ti

ti.init()


@ti.kernel
def p() -> ti.f32:
    print(42)
    return 40 + 2


print(p())
