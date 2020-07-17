import taichi as ti
try:
    import taichi_glsl as tl
except ImportError:
    print('This example needs the extension library Taichi GLSL to work.'
          'Please run `pip install --user taichi_glsl` to install it.')
ti.init(arch=ti.gpu)

N = 48
dt = 5e-5
dx = 1 / N
NF = (N, N, 2)
NV = (N + 1, N + 1)
E, nu = 3e5, 0.4
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
ball_pos, ball_radius = tl.vec(0.5, 0.0), 0.3
gravity = tl.vec(0, -40)
damping = 12.5

pos = ti.Vector.var(2, ti.f32, NV, needs_grad=True)
vel = ti.Vector.var(2, ti.f32, NV)
B = ti.Matrix.var(2, 2, ti.f32, NF)
F = ti.Matrix.var(2, 2, ti.f32, NF, needs_grad=True)
V = ti.var(ti.f32, NF)
phi = ti.var(ti.f32, NF)
U = ti.var(ti.f32, (), needs_grad=True)

@ti.func
def f2v(i):
    a, t = i.xy, i.z
    b, c, d = a + tl.D.x_, a + tl.D.xx, a + tl.D._x
    if t != 0:
        a = d
    return ti.static(a, b, c)

@ti.kernel
def update_U():
    for i in ti.grouped(B):
        ia, ib, ic = f2v(i)
        a, b, c = pos[ia], pos[ib], pos[ic]
        V[i] = abs((a - c).cross(b - c))
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]
    for i in ti.grouped(B):
        F_i = F[i]
        J_i = F_i.determinant()
        log_J_i = ti.log(F_i.determinant())
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
        phi_i += -mu * log_J_i + lam / 2 * log_J_i ** 2
        phi[i] = phi_i
        U[None] += V[i] * phi_i

@ti.kernel
def advance():
    for i in ti.grouped(pos):
        f_i = pos.grad[i] * (-1 / dx)
        vel[i] = vel[i] * ti.exp(-dt * damping) + dt * (f_i + gravity)
    for i in ti.grouped(pos):
        vel[i] = tl.ballBoundReflect(pos[i], vel[i], ball_pos, ball_radius)
        vel[i] = tl.boundReflect(pos[i], vel[i], 0, 1, 0)
        pos[i] = pos[i] + dt * vel[i]

@ti.kernel
def init():
    for i in ti.grouped(pos):
        pos[i] = i / N * 0.25 + tl.vec(0.45, 0.45)
        vel[i] = tl.vec2(0.0)
    for i in ti.grouped(B):
        ia, ib, ic = f2v(i)
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()

init()
gui = ti.GUI('FEM88')
while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'r':
            init()
    for i in range(50):
        with ti.Tape(loss=U):
            update_U()
        advance()
    gui.circles(pos.to_numpy().reshape((N + 1)**2, 2), radius=2, color=0xffaa33)
    gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
    gui.show()
