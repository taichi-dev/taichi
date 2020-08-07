import taichi as ti
import numpy as np
import numba

ti.init(arch=ti.opengl)

dim = 2
n_grid = 128
n_particles = n_grid ** dim // 2 ** (dim - 1)
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(dim, ti.f32, n_particles)
v = ti.Vector.field(dim, ti.f32, n_particles)
C = ti.Matrix.field(dim, dim, ti.f32, n_particles)
J = ti.field(ti.f32, n_particles)

grid_v = ti.Vector.field(dim, ti.f32, (n_grid,) * dim)
grid_m = ti.field(ti.f32, (n_grid,) * dim)

neighbour = (3,) * dim

@ti.kernel
def substep():
    for I in ti.grouped(grid_m):
        grid_v[I] = grid_v[I] * 0
        grid_m[I] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix.identity(ti.f32, dim) * stress + p_mass * C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
        grid_v[I][1] -= dt * gravity
        cond = I < bound and grid_v[I] < 0 or I > n_grid - bound and grid_v[I] > 0
        grid_v[I] = 0 if cond else grid_v[I]
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = v[p] * 0
        new_C = C[p] * 0
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.2
        J[i] = 1


@numba.njit
def field_edges(mass):
    begin, end = [], []

    def f(i, j):
        return mass[i, j] >= 1e-2

    n, m = mass.shape
    d, e = 1 / n, 1 / n
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            u, v = i * d, j * e
            if f(i, j) and not f(i, j + 1):
                begin.append((u, v + e))
                end.append((u + d, v + e))
            if f(i, j) and not f(i, j - 1):
                begin.append((u, v))
                end.append((u + d, v))
            if f(i, j) and not f(i - 1, j):
                begin.append((u, v))
                end.append((u, v + e))
            if f(i, j) and not f(i + 1, j):
                begin.append((u + d, v))
                end.append((u + d, v + e))

    return np.array(begin), np.array(end)


init()
gui = ti.GUI('MPM3D', background_color=0x112F41)
while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(10):
        substep()
    pos = x.to_numpy()
    mass = grid_m.to_numpy() * 1e5
    begin, end = field_edges(mass)
    gui.lines(begin, end)
    gui.circles(pos[:, :2], radius=1.5, color=0x068587)
    gui.show()
