import math

from utils import benchmark_async

import taichi as ti

# TODO: staggerred grid


@benchmark_async
def simple_advection(scale):
    n = 256 * 2**int((math.log(scale, 2)) // 2)
    x = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
    new_x = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
    v = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))
    dx = 1 / n
    inv_dx = 1 / dx
    dt = 0.01

    stagger = ti.Vector([0.5, 0.5])

    @ti.func
    def Vector2(x, y):
        return ti.Vector([x, y])

    @ti.kernel
    def init():
        for i, j in v:
            v[i, j] = ti.Vector([j / n - 0.5, 0.5 - i / n])

        for i, j in ti.ndrange(n * 4, n * 4):
            ret = ti.taichi_logo(ti.Vector([i, j]) / (n * 4))
            x[i // 4, j // 4][0] += ret / 16
            x[i // 4, j // 4][1] += ret / 16
            x[i // 4, j // 4][2] += ret / 16

    @ti.func
    def vec(x, y):
        return ti.Vector([x, y])

    @ti.func
    def clamp(p):
        for d in ti.static(range(p.n)):
            p[d] = min(1 - 1e-4 - dx + stagger[d] * dx,
                       max(p[d], stagger[d] * dx))
        return p

    @ti.func
    def sample_bilinear(x, p):
        p = clamp(p)

        p_grid = p * inv_dx - stagger

        I = ti.cast(ti.floor(p_grid), ti.i32)
        f = p_grid - I
        g = 1 - f

        return x[I] * (g[0] * g[1]) + x[I + vec(1, 0)] * (f[0] * g[1]) + x[
            I + vec(0, 1)] * (g[0] * f[1]) + x[I + vec(1, 1)] * (f[0] * f[1])

    @ti.func
    def velocity(p):
        return sample_bilinear(v, p)

    @ti.func
    def sample_min(x, p):
        p = clamp(p)
        p_grid = p * inv_dx - stagger
        I = ti.cast(ti.floor(p_grid), ti.i32)

        return min(x[I], x[I + vec(1, 0)], x[I + vec(0, 1)], x[I + vec(1, 1)])

    @ti.func
    def sample_max(x, p):
        p = clamp(p)
        p_grid = p * inv_dx - stagger
        I = ti.cast(ti.floor(p_grid), ti.i32)

        return max(x[I], x[I + vec(1, 0)], x[I + vec(0, 1)], x[I + vec(1, 1)])

    @ti.func
    def backtrace(I, dt):  # RK3
        p = (I + stagger) * dx
        v1 = velocity(p)
        p1 = p - 0.5 * dt * v1
        v2 = velocity(p1)
        p2 = p - 0.75 * dt * v2
        v3 = velocity(p2)
        p -= dt * (2 / 9 * v1 + 1 / 3 * v2 + 4 / 9 * v3)
        return p

    @ti.func
    def semi_lagrangian(x, new_x, dt):
        for I in ti.grouped(x):
            new_x[I] = sample_bilinear(x, backtrace(I, dt))

    @ti.kernel
    def advect():
        semi_lagrangian(x.get_scalar_field(0), new_x.get_scalar_field(0), dt)
        semi_lagrangian(x.get_scalar_field(1), new_x.get_scalar_field(1), dt)
        semi_lagrangian(x.get_scalar_field(2), new_x.get_scalar_field(2), dt)

        for I in ti.grouped(x):
            x[I] = new_x[I]

    init()

    def task():
        for i in range(10):
            advect()

    ti.benchmark(task, repeat=100)

    visualize = False

    if visualize:
        gui = ti.GUI('Advection schemes', (n, n))
        for i in range(10):
            for _ in range(10):
                advect()
            gui.set_image(x.to_numpy())
            gui.show()


if __name__ == '__main__':
    simple_advection()
