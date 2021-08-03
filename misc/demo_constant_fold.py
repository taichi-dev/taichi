# https://github.com/taichi-dev/taichi/pull/839#issuecomment-626217806
import taichi as ti

ti.init(print_ir=True)
#ti.core.toggle_advanced_optimization(False)


@ti.kernel
def calc_pi() -> ti.f32:
    term = 1.0
    sum = 0.0
    divisor = 1
    for i in ti.static(range(10)):
        sum += term / divisor
        term *= -1 / 3
        divisor += 2
    return sum * ti.sqrt(12.0)


print(calc_pi())
