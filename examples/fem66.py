# https://www.bilibili.com/video/BV1ZK411H7Hc?p=3
import taichi as ti
import taichi_glsl as tl  # a handy extension library
ti.init(arch=ti.gpu)

N = 40
dt = 6e-5
damping = 12.5
gravity = tl.vec(0, -40)
NF = (N, N, 2)  # faces tensor
NV = (N + 1, N + 1)  # vertices tensor
E, nu = 1e5, 0.4 # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
ball_pos, ball_radius = tl.vec(0.5, 0.0), 0.3  # the ball boundary
pos = ti.Vector.var(2, ti.f32, NV, needs_grad=True)
vel = ti.Vector.var(2, ti.f32, NV)
B = ti.Matrix.var(2, 2, ti.f32, NF)  # pre-computed rhs of F = D @ B
U = ti.var(ti.f32, (), needs_grad=True)  # total potential energy

@ti.func
def f2v(e):  # get 3 vertices around face e
    a, b, c, d = e.xy, e.xy + tl.D.x_, e.xy + tl.D.xx, e.xy + tl.D._x
    return ti.static(d if e.z else a, b, c)

@ti.kernel
def update_U():  # update total potential energy
    for e in ti.grouped(B):  # loop over all faces
        ia, ib, ic = f2v(e)  # get vertices around this face
        a_c, b_c = pos[ia] - pos[ic], pos[ib] - pos[ic]
        F = ti.Matrix.cols([a_c, b_c]) @ B[i]  # F = D @ B
        log_J = ti.log(F.determinant())   # J = jacobian deteriminant
        # Neo-Hookean according to: https://www.bilibili.com/video/BV1ZK411H7Hc?p=3
        phi = mu / 2 * ((F.T @ F).trace() - 2) + -mu * log_J + lam / 2 * log_J**2
        U[None] += abs(a_c.cross(b_c)) * phi  # intergrate over V_e * phi

@ti.kernel
def advance():  # update vertices' pos and vel
    for i in ti.grouped(pos):  # loop over all vertices, update vel
        force = pos.grad[i] * -N  # f_i = -dU / dx, here pos.grad is computed from U
        vel[i] = vel[i] * ti.exp(-dt * damping) + dt * (force + gravity)
    for i in ti.grouped(pos):  # do boundary condition and update pos
        vel[i] = tl.ballBoundReflect(pos[i], vel[i], ball_pos, ball_radius)
        vel[i] = tl.boundReflect(pos[i], vel[i], 0, 1, 0)  # rect boundary
        pos[i] = pos[i] + dt * vel[i]  # explicit-euler

@ti.kernel
def init():
    for i in ti.grouped(pos):
        pos[i] = i / N * 0.25 + 0.45
        vel[i] = tl.vec2(0.0)
    for e in ti.grouped(B):
        ia, ib, ic = f2v(e)
        a_c, b_c = pos[ia] - pos[ic], pos[ib] - pos[ic]
        B[e] = ti.Matrix.cols([a_c, b_c]).inverse()  # pre-compute rhs of F = D @ B

init()
gui = ti.GUI('FEM88')
while gui.running and not gui.get_event(gui.ESCAPE):
    for i in range(50):
        with ti.Tape(loss=U):  # from our accumation to U, obtain the gradient of U
            update_U()  # accumate to U so that pos.grad is computed by Taichi autodiff
        advance()  # step forward according to the computed U gradient
    # paint the material particles:
    gui.circles(pos.to_numpy().reshape((N + 1)**2, 2), radius=2, color=0xffaa33)
    gui.circle(ball_pos, radius=ball_radius * 512, color=0x666666)
    gui.show()
