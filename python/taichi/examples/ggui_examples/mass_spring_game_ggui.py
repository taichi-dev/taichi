import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

spring_Y = ti.field(dtype=ti.f32, shape=())  # Young's modulus
paused = ti.field(dtype=ti.i32, shape=())
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())

max_num_particles = 1024
particle_mass = 1.0
dt = 1e-3
substeps = 10

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
fixed = ti.field(dtype=ti.i32, shape=max_num_particles)

indices = ti.field(dtype=ti.i32,
                   shape=max_num_particles * max_num_particles * 2)
per_vertex_color = ti.Vector.field(3, ti.f32, shape=max_num_particles)

# rest_length[i, j] == 0 means i and j are NOT connected
rest_length = ti.field(dtype=ti.f32,
                       shape=(max_num_particles, max_num_particles))


@ti.kernel
def substep():
    n = num_particles[None]

    # Compute force
    for i in range(n):
        # Gravity
        f[i] = ti.Vector([0, -9.8]) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()

                # Spring force
                f[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i, j] -
                                           1) * d

                # Dashpot damping
                v_rel = (v[i] - v[j]).dot(d)
                f[i] += -dashpot_damping[None] * v_rel * d

    # We use a semi-implicit Euler (aka symplectic Euler) time integrator
    for i in range(n):
        if not fixed[i]:
            v[i] += dt * f[i] / particle_mass
            v[i] *= ti.exp(-dt * drag_damping[None])  # Drag damping

            x[i] += v[i] * dt
        else:
            v[i] = ti.Vector([0, 0])

        # Collide with four walls
        for d in ti.static(range(2)):
            # d = 0: treating X (horizontal) component
            # d = 1: treating Y (vertical) component

            if x[i][d] < 0:  # Bottom and left
                x[i][d] = 0  # move particle inside
                v[i][d] = 0  # stop it from moving further

            if x[i][d] > 1:  # Top and right
                x[i][d] = 1  # move particle inside
                v[i][d] = 0  # stop it from moving further


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    # Taichi doesn't support using vectors as kernel arguments yet, so we pass scalars
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    fixed[new_particle_id] = fixed_
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        connection_radius = 0.15
        if dist < connection_radius:
            # Connect the new particle with particle i
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(num_particles[None]):
        p = ti.Vector([pos_x, pos_y])
        v[i] += -dt * substeps * (x[i] - p) * 100


@ti.kernel
def render():
    for i in indices:
        indices[i] = max_num_particles - 1
    n = num_particles[None]
    for i in range(n + 1, max_num_particles):
        x[i] = ti.Vector([-1, -1])  # hide them
    for i in range(n):
        if fixed[i]:
            per_vertex_color[i] = ti.Vector([1, 0, 0])
        else:
            per_vertex_color[i] = ti.Vector([0, 0, 0])
    for i in range(n):
        for j in range(i + 1, n):
            line_id = i * max_num_particles + j
            if rest_length[i, j] != 0:
                indices[line_id * 2] = i
                indices[line_id * 2 + 1] = j


def main():
    window = ti.ui.Window('Explicit Mass Spring System', (768, 768),
                          vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    spring_Y[None] = 1000
    drag_damping[None] = 1
    dashpot_damping[None] = 100

    new_particle(0.3, 0.3, False)
    new_particle(0.3, 0.4, False)
    new_particle(0.4, 0.4, False)

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            elif e.key == ti.ui.SPACE:
                paused[None] = not paused[None]
            elif e.key == ti.ui.LMB:
                pos = window.get_cursor_pos()
                new_particle(pos[0], pos[1],
                             int(window.is_pressed(ti.ui.SHIFT)))
            elif e.key == 'c':
                num_particles[None] = 0
                rest_length.fill(0)

        if window.is_pressed(ti.ui.RMB):
            cursor_pos = window.get_cursor_pos()
            attract(cursor_pos[0], cursor_pos[1])

        if not paused[None]:
            for step in range(substeps):
                substep()

        render()
        canvas.set_background_color((1, 1, 1))
        canvas.lines(x, indices=indices, color=(0, 0, 0), width=0.01)
        canvas.circles(x, per_vertex_color=per_vertex_color, radius=0.02)

        with gui.sub_window("mass spring", 0.05, 0.05, 0.9, 0.2) as w:
            w.text(
                "Left click: add mass point (with shift to fix); Right click: attract"
            )
            w.text("C: clear all; Space: pause")
            spring_Y[None] = w.slider_float("Spring Young's modulus",
                                            spring_Y[None], 100, 10000)
            drag_damping[None] = w.slider_float("Drag damping",
                                                drag_damping[None], 0.0, 10)
            dashpot_damping[None] = w.slider_float("Dashpot damping",
                                                   dashpot_damping[None], 10,
                                                   1000)

        window.show()


if __name__ == '__main__':
    main()
