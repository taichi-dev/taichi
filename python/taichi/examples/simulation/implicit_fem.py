import argparse

import numpy as np
from taichi._lib import core as _ti_core

import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument('--exp',
                    choices=['implicit', 'explicit'],
                    default='implicit')
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--gui', choices=['auto', 'ggui', 'cpu'], default='auto')
parser.add_argument('place_holder', nargs='*')
args = parser.parse_args()

ti.init(arch=ti.gpu, dynamic_index=True)

if args.gui == 'auto':
    if _ti_core.GGUI_AVAILABLE:
        args.gui = 'ggui'
    else:
        args.gui = 'cpu'

E, nu = 5e4, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # lambda = 0
density = 1000.0
dt = 2e-4

if args.exp == 'implicit':
    dt = 1e-2

n_cube = np.array([5] * 3)
n_verts = np.product(n_cube)
n_cells = 5 * np.product(n_cube - 1)
dx = 1 / (n_cube.max() - 1)

vertices = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)

x = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
ox = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
v = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
f = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
mul_ans = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
m = ti.field(dtype=ti.f32, shape=n_verts)

n_cells = (n_cube - 1).prod() * 5
B = ti.Matrix.field(args.dim, args.dim, dtype=ti.f32, shape=n_cells)
W = ti.field(dtype=ti.f32, shape=n_cells)


@ti.func
def i2p(I):
    return (I.x * n_cube[1] + I.y) * n_cube[2] + I.z


@ti.func
def set_element(e, I, verts):
    for i in ti.static(range(args.dim + 1)):
        vertices[e][i] = i2p(I + (([verts[i] >> k for k in range(3)] ^ I) & 1))


@ti.kernel
def get_vertices():
    '''
    This kernel partitions the cube into tetrahedrons.
    Each unit cube is divided into 5 tetrahedrons.
    '''
    for I in ti.grouped(ti.ndrange(*(n_cube - 1))):
        e = ((I.x * (n_cube[1] - 1) + I.y) * (n_cube[2] - 1) + I.z) * 5
        for i, j in ti.static(enumerate([0, 3, 5, 6])):
            set_element(e + i, I, (j, j ^ 1, j ^ 2, j ^ 4))
        set_element(e + 4, I, (1, 2, 4, 7))
    for I in ti.grouped(ti.ndrange(*(n_cube))):
        ox[i2p(I)] = I * dx


@ti.func
def Ds(verts):
    return ti.Matrix.cols([x[verts[i]] - x[verts[3]] for i in range(3)])


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def get_force_func(c, verts):
    F = Ds(verts) @ B[c]
    P = ti.Matrix.zero(ti.f32, 3, 3)
    U, sig, V = ssvd(F)
    P = 2 * mu * (F - U @ V.transpose())
    H = -W[c] * P @ B[c].transpose()
    for i in ti.static(range(3)):
        force = ti.Vector([H[j, i] for j in range(3)])
        f[verts[i]] += force
        f[verts[3]] -= force


@ti.kernel
def get_force():
    for c in vertices:
        get_force_func(c, vertices[c])
    for u in f:
        f[u].y -= 9.8 * m[u]


@ti.kernel
def matmul_cell(ret: ti.template(), vel: ti.template()):
    for i in ret:
        ret[i] = vel[i] * m[i]
    for c in vertices:
        verts = vertices[c]
        W_c = W[c]
        B_c = B[c]
        for u in range(4):
            for d in range(3):
                dD = ti.Matrix.zero(ti.f32, 3, 3)
                if u == 3:
                    for j in range(3):
                        dD[d, j] = -1
                else:
                    dD[d, u] = 1
                dF = dD @ B_c
                dP = 2.0 * mu * dF
                dH = -W_c * dP @ B_c.transpose()
                for i in range(3):
                    for j in range(3):
                        tmp = (vel[verts[i]][j] - vel[verts[3]][j])
                        ret[verts[u]][d] += -dt**2 * dH[j, i] * tmp


@ti.kernel
def add(ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
    for i in ans:
        ans[i] = a[i] + k * b[i]


@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> ti.f32:
    ans = 0.0
    for i in a:
        ans += a[i].dot(b[i])
    return ans


b = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
r0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
p0 = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)


@ti.kernel
def get_b():
    for i in b:
        b[i] = m[i] * v[i] + dt * f[i]


def cg():
    def mul(x):
        matmul_cell(mul_ans, x)
        return mul_ans

    get_force()
    get_b()
    mul(v)
    add(r0, b, -1, mul(v))

    d = p0
    d.copy_from(r0)
    r_2 = dot(r0, r0)
    n_iter = 50
    epsilon = 1e-6
    r_2_init = r_2
    r_2_new = r_2
    for iter in range(n_iter):
        q = mul(d)
        alpha = r_2_new / dot(d, q)
        add(v, v, alpha, d)
        add(r0, r0, -alpha, q)
        r_2 = r_2_new
        r_2_new = dot(r0, r0)
        if r_2_new <= r_2_init * epsilon**2: break
        beta = r_2_new / r_2
        add(d, r0, beta, d)
    f.fill(0)
    add(x, x, dt, v)


@ti.kernel
def advect():
    for p in x:
        v[p] += dt * (f[p] / m[p])
        x[p] += dt * v[p]
        f[p] = ti.Vector([0, 0, 0])


@ti.kernel
def init():
    for u in x:
        x[u] = ox[u]
        v[u] = [0.0] * 3
        f[u] = [0.0] * 3
        m[u] = 0.0
    for c in vertices:
        F = Ds(vertices[c])
        B[c] = F.inverse()
        W[c] = ti.abs(F.determinant()) / 6
        for i in range(4):
            m[vertices[c][i]] += W[c] / 4 * density
    for u in x:
        x[u].y += 1.0


@ti.kernel
def floor_bound():
    for u in x:
        if x[u].y < 0:
            x[u].y = 0
            if v[u].y < 0:
                v[u].y = 0


@ti.func
def check(u):
    ans = 0
    rest = u
    for i in ti.static(range(3)):
        k = rest % n_cube[2 - i]
        rest = rest // n_cube[2 - i]
        if k == 0: ans |= (1 << (i * 2))
        if k == n_cube[2 - i] - 1: ans |= (1 << (i * 2 + 1))
    return ans


su = 0
for i in range(3):
    su += (n_cube[i] - 1) * (n_cube[(i + 1) % 3] - 1)
indices = ti.field(ti.i32, shape=2 * su * 2 * 3)


@ti.kernel
def get_indices():
    # calculate all the meshes on surface
    cnt = 0
    for c in vertices:
        if c % 5 != 4:
            for i in ti.static([0, 2, 3]):
                verts = [vertices[c][(i + j) % 4] for j in range(3)]
                sum = check(verts[0]) & check(verts[1]) & check(verts[2])
                if sum:
                    m = ti.atomic_add(cnt, 1)
                    det = ti.Matrix.rows([
                        x[verts[i]] - [0.5, 1.5, 0.5] for i in range(3)
                    ]).determinant()
                    if det < 0:
                        tmp = verts[1]
                        verts[1] = verts[2]
                        verts[2] = tmp
                    indices[m * 3] = verts[0]
                    indices[m * 3 + 1] = verts[1]
                    indices[m * 3 + 2] = verts[2]


def substep():
    if args.exp == 'explicit':
        for i in range(10):
            get_force()
            advect()
    else:
        for i in range(1):
            cg()
    floor_bound()


if __name__ == '__main__':
    get_vertices()
    init()
    get_indices()

    if args.gui == 'ggui':
        res = (800, 600)
        window = ti.ui.Window("Implicit FEM", res, vsync=True)

        frame_id = 0
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.make_camera()
        camera.position(2.0, 2.0, 3.95)
        camera.lookat(0.5, 0.5, 0.5)
        camera.fov(55)

        def render():
            camera.track_user_inputs(window,
                                     movement_speed=0.03,
                                     hold_key=ti.ui.RMB)
            scene.set_camera(camera)

            scene.ambient_light((0.1, ) * 3)

            scene.point_light(pos=(0.5, 10.0, 0.5), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(10.0, 10.0, 10.0), color=(0.5, 0.5, 0.5))

            scene.mesh(x, indices, color=(0.73, 0.33, 0.23))

            canvas.scene(scene)

        while window.running:
            frame_id += 1
            frame_id = frame_id % 256
            substep()
            if window.is_pressed('r'):
                init()
            if window.is_pressed(ti.GUI.ESCAPE):
                break

            render()

            window.show()

    else:

        def T(a):

            phi, theta = np.radians(28), np.radians(32)

            a = a - 0.2
            x, y, z = a[:, 0], a[:, 1], a[:, 2]
            c, s = np.cos(phi), np.sin(phi)
            C, S = np.cos(theta), np.sin(theta)
            x, z = x * c + z * s, z * c - x * s
            u, v = x, y * C + z * S
            return np.array([u, v]).swapaxes(0, 1) + 0.5

        gui = ti.GUI('Implicit FEM')
        while gui.running:
            substep()
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
            if gui.is_pressed('r'):
                init()
            gui.clear(0x000000)
            gui.circles(T(x.to_numpy() / 3), radius=1.5, color=0xba543a)
            gui.show()
