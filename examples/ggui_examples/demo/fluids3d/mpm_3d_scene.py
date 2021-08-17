import numpy as np

import taichi as ti

ti.init(ti.cuda)

#dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
#dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
#dim, n_grid, steps, dt = 3, 128, 5, 1e-4

n_particles = n_grid**dim // 2**(dim - 1)

print(n_particles)

dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(dim, float, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
J = ti.field(float, n_particles)

colors = ti.Vector.field(3, float, n_particles)

grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
grid_m = ti.field(float, (n_grid, ) * dim)

neighbour = (3, ) * dim


@ti.kernel
def substep():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
    ti.block_dim(n_grid)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix.identity(float, dim) * stress + p_mass * C[p]
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
        cond = I < bound and grid_v[I] < 0 or I > n_grid - bound and grid_v[
            I] > 0
        grid_v[I] = 0 if cond else grid_v[I]
    ti.block_dim(n_grid)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])
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
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.15
        J[i] = 1
        colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])


init()

res = (1920, 1080)
window = ti.ui.Window("heyy", res)

frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

show_particles = True
camera_x = 2.0
camera_y = 2.0
camera_z = 2.0

use_random_colors = False
particles_color = (0, 0, 1)
particles_radius = 0.05

while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    for s in range(steps):
        substep()

    camera.position(camera_x, camera_y, camera_z)
    camera.lookat(0, 0, 0)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.ambient_light((0, 0, 0))
    if show_particles:
        if use_random_colors:
            scene.particles(x,
                            per_vertex_color=colors,
                            radius=particles_radius)
        else:
            scene.particles(x, color=particles_color, radius=particles_radius)
    scene.point_light(pos=(camera_x, camera_y, camera_z), color=(1, 1, 1))

    canvas.scene(scene)

    window.GUI.begin("Real MPM 3D", 0.1, 0.1, 0.2, 0.8)
    window.GUI.text("hello text")
    show_particles = window.GUI.checkbox("show particles", show_particles)
    camera_x = window.GUI.slider_float("camera x", camera_x, -10, 10)
    camera_y = window.GUI.slider_float("camera y", camera_y, -10, 10)
    camera_z = window.GUI.slider_float("camera z", camera_z, -10, 10)
    use_random_colors = window.GUI.checkbox("use_random_colors",
                                            use_random_colors)
    if not use_random_colors:
        particles_color = window.GUI.color_edit_3("particles color",
                                                  particles_color)
    particles_radius = window.GUI.slider_float("particles radius ",
                                               particles_radius, 0, 0.1)
    if window.GUI.button("restart"):
        init()
    window.GUI.end()

    #
    window.show()
