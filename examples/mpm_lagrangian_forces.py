import taichi as ti

ti.init(arch=ti.gpu)

real = ti.f32
dim = 2
n_particle_x = 100
n_particle_y = 8
n_particles = n_particle_x * n_particle_y
n_elements = (n_particle_x - 1) * (n_particle_y - 1) * 2
n_grid = 64
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-4
p_mass = 1
p_vol = 1
mu = 1
la = 1

x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles, needs_grad=True)
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
restT = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles, needs_grad=True)
total_energy = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
vertices = ti.field(dtype=ti.i32, shape=(n_elements, 3))


@ti.func
def compute_T(i):
    a = vertices[i, 0]
    b = vertices[i, 1]
    c = vertices[i, 2]
    ab = x[b] - x[a]
    ac = x[c] - x[a]
    return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])


@ti.kernel
def compute_rest_T():
    for i in range(n_elements):
        restT[i] = compute_T(i)


@ti.kernel
def compute_total_energy():
    for i in range(n_elements):
        currentT = compute_T(i)
        F = currentT @ restT[i].inverse()
        # NeoHookean
        I1 = (F @ F.transpose()).trace()
        J = F.determinant()
        element_energy = 0.5 * mu * (
            I1 - 2) - mu * ti.log(J) + 0.5 * la * ti.log(J)**2
        ti.atomic_add(total_energy[None], element_energy * 1e-3)


@ti.kernel
def p2g():
    for p in x:
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        affine = p_mass * C[p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                I = ti.Vector([i, j])
                dpos = (float(I) - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base + I] += weight * (p_mass * v[p] - x.grad[p] +
                                              affine @ dpos)
                grid_m[base + I] += weight * p_mass


bound = 3


@ti.kernel
def grid_op():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]
            grid_v[i, j].y -= dt * 9.8

            # center collision circle
            dist = ti.Vector([i * dx - 0.5, j * dx - 0.5])
            if dist.norm_sqr() < 0.005:
                dist = dist.normalized()
                grid_v[i, j] -= dist * min(0, grid_v[i, j].dot(dist))

            # box
            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0
            if i > n_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0
            if j > n_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0


@ti.kernel
def g2p():
    for p in x:
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
        fx = x[p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                I = ti.Vector([i, j])
                dpos = float(I) - fx
                g_v = grid_v[base + I]
                weight = w[i].x * w[j].y
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[p] = new_v
        x[p] += dt * v[p]
        C[p] = new_C


gui = ti.GUI("MPM", (640, 640), background_color=0x112F41)

mesh = lambda i, j: i * n_particle_y + j


def main():
    for i in range(n_particle_x):
        for j in range(n_particle_y):
            t = mesh(i, j)
            x[t] = [0.1 + i * dx * 0.5, 0.7 + j * dx * 0.5]
            v[t] = [0, -1]

    # build mesh
    for i in range(n_particle_x - 1):
        for j in range(n_particle_y - 1):
            # element id
            eid = (i * (n_particle_y - 1) + j) * 2
            vertices[eid, 0] = mesh(i, j)
            vertices[eid, 1] = mesh(i + 1, j)
            vertices[eid, 2] = mesh(i, j + 1)

            eid = (i * (n_particle_y - 1) + j) * 2 + 1
            vertices[eid, 0] = mesh(i, j + 1)
            vertices[eid, 1] = mesh(i + 1, j + 1)
            vertices[eid, 2] = mesh(i + 1, j)

    compute_rest_T()

    vertices_ = vertices.to_numpy()

    for f in range(600):
        for s in range(50):
            grid_m.fill(0)
            grid_v.fill(0)
            # Note that we are now differentiating the total energy w.r.t. the particle position.
            # Recall that F = - \partial (total_energy) / \partial x
            with ti.Tape(total_energy):
                # Do the forward computation of total energy and backward propagation for x.grad, which is later used in p2g
                compute_total_energy()
                # It's OK not to use the computed total_energy at all, since we only need x.grad
            p2g()
            grid_op()
            g2p()

        gui.circle((0.5, 0.5), radius=45, color=0x068587)
        # TODO: why is visualization so slow?
        particle_pos = x.to_numpy()
        for i in range(n_elements):
            for j in range(3):
                a, b = vertices_[i, j], vertices_[i, (j + 1) % 3]
                gui.line((particle_pos[a][0], particle_pos[a][1]),
                         (particle_pos[b][0], particle_pos[b][1]),
                         radius=1,
                         color=0x4FB99F)
        gui.circles(particle_pos, radius=1.5, color=0xF2B134)
        gui.line((0.00, 0.03), (1.0, 0.03), color=0xFFFFFF, radius=3)
        gui.show()


if __name__ == '__main__':
    main()
