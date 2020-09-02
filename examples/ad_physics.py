import taichi as ti
import math


dt = 0.02
times = 12
steps = 7
times = 36
speed = 0.1
gravity = 0.0006
learn = 0.01
N = 512


pos = ti.Vector.field(2, float, (), needs_grad=True)
vel = ti.Vector.field(2, float, (), needs_grad=True)
init_vel = ti.Vector.field(2, float, (), needs_grad=True)
loss = ti.field(float, (), needs_grad=True)
target = ti.Vector.field(2, float, ())
center = ti.Vector.field(2, float, ())


@ti.kernel
def substep():
    pos[None] += vel[None] * dt
    r = center[None] - pos[None]
    vel[None] += r / r.norm(3e-3)**3 * gravity * dt


@ti.kernel
def compute_loss():
    loss[None] = (pos[None] - target[None]).norm_sqr()


@ti.kernel
def init():
    pos[None] = [0.5, 0.5]
    vel[None] = init_vel[None]


@ti.kernel
def optimize():
    init_vel[None] -= learn * (0.4 + ti.random() * 0.8) * init_vel.grad[None]


target[None] = [0.75, 0.65]
center[None] = [0.6, 0.6]
init_vel[None] = [0.1, -0.1]
gui = ti.GUI()

while gui.running:

    if gui.get_event(gui.LMB, gui.PRESS):
        target[None] = gui.get_cursor_pos()

    with ti.Tape(loss):
        init()
        for t in range(times):
            for s in range(steps):
                substep()
            gui.circle(pos[None].value, color=0xffcc66, radius=6)
            gui.circle(target[None].value, color=0xcc66ff, radius=6)
            gui.circle(center[None].value, color=0x666666, radius=10)
            gui.show()

        compute_loss()

    optimize()
