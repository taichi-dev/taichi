import taichi as ti
ti.init(arch=ti.cpu)

n_particles = 8192 * 3
n_grid = 512
dx = 1 / n_grid
dt = 5e-5

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(2, ti.f32, n_particles)
v = ti.Vector.field(2, ti.f32, n_particles)
C = ti.Matrix.field(2, 2, ti.f32, n_particles)
J = ti.field(ti.f32, n_particles)

block_size = 16
assert n_grid % block_size == 0

block0 = ti.root.pointer(ti.ij, n_grid // block_size)
block1 = block0.dense(ti.ij, block_size)

grid_v = ti.Vector.field(2, ti.f32)
grid_m = ti.field(ti.f32)

block1.place(grid_v, grid_m)

background = ti.field(ti.f32, shape=(n_grid, n_grid))


@ti.kernel
def substep():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j][1] -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

    for i, j in background:
        if ti.is_active(block0, [i, j]):
            background[i, j] = 1.0
        else:
            background[i, j] = 0.8


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.25 + 0.2, ti.random() * 0.25 + 0.2]
        v[i] = [0, -1]
        J[i] = 1


init()
gui = ti.GUI('MLS-MPM2D Sparse', res=n_grid)
while gui.running and not gui.get_event(gui.PRESS):
    for s in range(20):
        background.fill(0)
        block0.deactivate_all()
        substep()
    gui.set_image(background)
    gui.circles(x.to_numpy(), radius=0.75, color=0x333333)
    gui.show()
