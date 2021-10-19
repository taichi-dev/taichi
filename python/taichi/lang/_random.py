import math

from taichi.lang.kernel_impl import func

import taichi as ti


@func
def randn(dt):
    '''
    Generates a random number from standard normal distribution
    using the Box-Muller transform.
    '''
    assert dt == ti.f32 or dt == ti.f64
    u1 = ti.random(dt)
    u2 = ti.random(dt)
    r = ti.sqrt(-2 * ti.log(u1))
    c = ti.cos(math.tau * u2)
    return r * c
