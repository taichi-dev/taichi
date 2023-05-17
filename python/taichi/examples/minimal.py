import taichi as ti

ti.init(print_ir=True)


@ti.kernel
def p() -> ti.f32:
    y = 0.0
    x = 1
    y -= not x
    return y


print(p())
