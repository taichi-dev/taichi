# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti

ti.init(arch=ti.vulkan)

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

pos = ti.Vector.ndarray(12, ti.f32, n_particles)
x = ti.Vector.field(2, ti.f32, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


@ti.kernel
def substep(pos: ti.any_arr(element_dim=1)):
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
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
        grid_v[i, j].y -= dt * gravity
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
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        # VBO Attributes: all (pos, normal, tex, color)
        pos[p] = [x[p][0], x[p][1], 0,0,0,0,0,0,0,0,0,0]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def init(pos: ti.any_arr(element_dim=1)):
    for i in range(n_particles):
        #pos[i] = [x[i][0], x[i][1], 0,0,0,0,0,0,0,0,0,0]
        pos[i][0] = ti.random() * 0.4 + 0.2
        pos[i][1] = ti.random() * 0.4 + 0.2
        x[i] = [pos[i][0], pos[i][1]]
        v[i] = [0, -1]
        J[i] = 1

#init(pos)
#gui = ti.GUI('MPM88')
#while gui.running and not gui.get_event(gui.ESCAPE):
#    for s in range(50):
#        substep(pos)
#    gui.clear(0x112F41)
#    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
#    gui.show()

m = ti.aot.Module(ti.vulkan)
m.add_kernel(init, (pos,))
m.add_kernel(substep, (pos,))
m.save('.', 'mpm88')
