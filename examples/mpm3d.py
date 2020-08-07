import taichi as ti
import numpy as np
import numba

ti.init(arch=ti.opengl)

dim = 3
n_grid = 32
n_particles = n_grid ** dim // 2 ** (dim - 1)
dx = 1 / n_grid
dt = 4e-4

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

    def a(i, j):
        return mass[i, j] >= 1e-2

    n, m = mass.shape
    d, e = 1 / n, 1 / n
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            u, v = i * d, j * e
            if a(i, j) and not a(i, j + 1):
                begin.append((u, v + e))
                end.append((u + d, v + e))
            if a(i, j) and not a(i, j - 1):
                begin.append((u, v))
                end.append((u + d, v))
            if a(i, j) and not a(i - 1, j):
                begin.append((u, v))
                end.append((u, v + e))
            if a(i, j) and not a(i + 1, j):
                begin.append((u + d, v))
                end.append((u + d, v + e))

    return np.array(begin), np.array(end)


@numba.njit
def field_edges3(mass):
    A, B = [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]
    C, D = [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]

    def a(i, j, k):
        return mass[i, j, k] >= 1e-2

    n, m, o = mass.shape
    d, e, f = 1 / n, 1 / m, 1 / o
    for i in range(1, n):
        for j in range(1, m):
            for k in range(1, o):
                if not a(i, j, k):
                    continue

                u, v, w = i * d, j * e, k * f
                if not a(i, j + 1, k):
                    A.append((u, v + e, w))
                    B.append((u, v + e, w + f))
                    C.append((u + d, v + e, w + f))
                    D.append((u + d, v + e, w))
                if not a(i, j - 1, k):
                    A.append((u, v, w))
                    B.append((u, v, w + f))
                    C.append((u + d, v, w + f))
                    D.append((u + d, v, w))
                if not a(i - 1, j, k):
                    A.append((u, v, w))
                    B.append((u, v + e, w))
                    C.append((u, v + e, w + f))
                    D.append((u, v, w + f))
                if not a(i + 1, j, k):
                    A.append((u + d, v, w))
                    B.append((u + d, v + e, w))
                    C.append((u + d, v + e, w + f))
                    D.append((u + d, v, w + f))
                if not a(i, j, k - 1):
                    A.append((u, v, w))
                    B.append((u, v + e, w))
                    C.append((u + d, v + e, w))
                    D.append((u + d, v, w))
                if not a(i, j, k + 1):
                    A.append((u, v, w + f))
                    B.append((u, v + e, w + f))
                    C.append((u + d, v + e, w + f))
                    D.append((u + d, v, w + f))

    return np.array(A), np.array(B), np.array(C), np.array(D)


@numba.njit
def Tker(a):
    a = a - 0.5

    x, y, z = a[:, 0], a[:, 1], a[:, 2]

    phi = np.radians(28)
    theta = np.radians(32)

    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)

    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S

    return u + 0.5, v + 0.5


def T(a):
    if dim == 2:
        return a

    u, v = Tker(a)
    return np.array([u, v]).swapaxes(0, 1)


init()
gui = ti.GUI('MPM3D', background_color=0x112F41)
while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(15):
        substep()
    pos = x.to_numpy()

    if 0:
        mass = grid_m.to_numpy() * 1e5
        if dim == 2:
            begin, end = field_edges(mass)
            gui.lines(begin, end)

        else:
            A, B, C, D = field_edges3(mass)
            num = A.shape[0]

            writer = ti.PLYWriter(num_vertices=num * 4,
                    num_faces=A.shape[0], face_type='quad')
            V = np.concatenate([A, B, C, D], axis=0)
            writer.add_vertex_pos(V[:, 0], V[:, 1], V[:, 2])

            F = [[j * num + i for j in range(4)] for i in range(num)]
            writer.add_faces(np.array(F).reshape(num * 4))

            writer.export_frame(gui.frame, '/tmp/mpm3d.ply')

    gui.circles(T(pos), radius=2, color=0x66ccff)
    gui.show()
