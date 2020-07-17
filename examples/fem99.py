import taichi as ti
try:
    import taichi_glsl as tl
except ImportError:
    print('This example needs the extension library Taichi GLSL to work.'
          'Please run `pip install --user taichi_glsl` to install it.')
ti.init(arch=ti.gpu)

N = 32
dt = 1e-4
dx = 1 / N
NF = 2 * N ** 2
NV = (N + 1) ** 2
E, nu = 4e4, 0.2
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
ball_pos, ball_radius = tl.vec(0.5, 0.0), 0.32
gravity = tl.vec(0, -40)
damping = 12.5

pos = ti.Vector.var(2, ti.f32, NV, needs_grad=True)
vel = ti.Vector.var(2, ti.f32, NV)
f2v = ti.Vector.var(3, ti.i32, NF)
B = ti.Matrix.var(2, 2, ti.f32, NF)
F = ti.Matrix.var(2, 2, ti.f32, NF, needs_grad=True)
V = ti.var(ti.f32, NF)
phi = ti.var(ti.f32, NF)
U = ti.var(ti.f32, (), needs_grad=True)

@ti.kernel
def update_U():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a = pos[ia]
        b = pos[ib]
        c = pos[ic]
        V[i] = abs((a - c).cross(b - c))
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]
    for i in range(NF):
        F_i = F[i]
        J_i = F_i.determinant()
        log_J_i = ti.log(J_i)
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
        phi_i -= mu * log_J_i
        phi_i += lam / 2 * log_J_i ** 2
        phi[i] = phi_i
        U[None] += V[i] * phi_i

@ti.kernel
def advance():
    for i in range(NV):
        f_i = -pos.grad[i] * (1 / dx)
        vel[i] += dt * (f_i + gravity)
        vel[i] *= ti.exp(-dt * damping)
    for i in range(NV):
        vel[i] = tl.ballBoundReflect(pos[i], vel[i], ball_pos, ball_radius)
        vel[i] = tl.boundReflect(pos[i], vel[i], 0, 1, 0)
        pos[i] += dt * vel[i]

@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = tl.vec(i, j) / N * 0.25 + tl.vec(0.45, 0.45)
        vel[k] = tl.vec2(0.0)
    for i in range(NF):
        a = pos[f2v[i].x]
        b = pos[f2v[i].y]
        c = pos[f2v[i].z]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()

@ti.kernel
def init_geo():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]

init_geo()
init_pos()
gui = ti.GUI('FEM99')
while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'r':
            init_pos()
    for i in range(30):
        with ti.Tape(loss=U):
            update_U()
        advance()
    gui.circles(pos.to_numpy(), radius=2, color=0xffaa33)
    gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
    gui.show()
