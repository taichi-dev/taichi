import taichi as ti

ti.init(arch=ti.gpu)

use_mc = True
mc_clipping = False
pause = False

# Runge-Kutta order
rk = 3

n = 512
x = ti.field(ti.f32, shape=(n, n))
new_x = ti.field(ti.f32, shape=(n, n))
new_x_aux = ti.field(ti.f32, shape=(n, n))
v = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))
dx = 1 / n
inv_dx = 1 / dx
dt = 0.05

stagger = ti.Vector([0.5, 0.5])


@ti.func
def Vector2(x, y):
    return ti.Vector([x, y])


@ti.kernel
def paint():
    for i, j in ti.ndrange(n * 4, n * 4):
        ret = ti.taichi_logo(ti.Vector([i, j]) / (n * 4))
        x[i // 4, j // 4] += ret / 16


@ti.kernel
def init_v():
    for i, j in v:
        v[i, j] = ti.Vector([j / n - 0.5, 0.5 - i / n])


@ti.func
def vec(x, y):
    return ti.Vector([x, y])

@ti.func
def clamp(p):
    for d in ti.static(range(p.n)):
        p[d] = min(1 - 1e-4 - dx + stagger[d] * dx, max(p[d], stagger[d] * dx))
    return p

@ti.func
def sample_bilinear(x, p):
    p = clamp(p)
    
    p_grid = p * inv_dx - stagger
    
    I = ti.cast(ti.floor(p_grid), ti.i32)
    f = p_grid - I
    g = 1 - f
    
    return x[I] * (g[0] * g[1]) + x[I + vec(1, 0)] * (
            f[0] * g[1]) + x[I + vec(0, 1)] * (
                   g[0] * f[1]) + x[I + vec(1, 1)] * (f[0] * f[1])

@ti.func
def velocity(p):
    return sample_bilinear(v, p)

@ti.func
def sample_min(x, p):
    p = clamp(p)
    p_grid = p * inv_dx - stagger
    I = ti.cast(ti.floor(p_grid), ti.i32)
    
    return min(x[I],  x[I + vec(1, 0)], x[I + vec(0, 1)], x[I + vec(1, 1)])

@ti.func
def sample_max(x, p):
    p = clamp(p)
    p_grid = p * inv_dx - stagger
    I = ti.cast(ti.floor(p_grid), ti.i32)
    
    return max(x[I],  x[I + vec(1, 0)], x[I + vec(0, 1)], x[I + vec(1, 1)])

@ti.func
def backtrace(I, dt):
    p = (I + stagger) * dx
    if ti.static(rk == 1):
        p -= dt * velocity(p)
    elif ti.static(rk == 2):
        p_mid = p - 0.5 * dt * velocity(p)
        p -= dt * velocity(p_mid)
    elif ti.static(rk == 3):
        v1 = velocity(p)
        p1 = p - 0.5 * dt * v1
        v2 = velocity(p1)
        p2 = p - 0.75 * dt * v2
        v3 = velocity(p2)
        p -= dt * (2 / 9 * v1 + 1 / 3 * v2 + 4 / 9 * v3)
    else:
        ti.static_print(f"RK{rk} is not supported.")
    
    return p

@ti.func
def semi_lagrangian(x, new_x, dt):
    # Note: this loop is parallelized
    for I in ti.grouped(x):
        new_x[I] = sample_bilinear(x, backtrace(I, dt))

@ti.kernel
def advect():
    semi_lagrangian(x, new_x, dt)
    
    for I in ti.grouped(x):
        x[I] = new_x[I]

init_v()

paint()

gui = ti.GUI('Advection schemes', (512, 512))

while True:
    for i in range(1):
        advect()
    gui.set_image(x.to_numpy())
    gui.show()
