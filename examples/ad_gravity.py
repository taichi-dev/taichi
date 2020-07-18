import taichi as ti
ti.init()

N = 8
dt = 5e-5

pos = ti.Vector.var(2, ti.f32, N, needs_grad=True)
vel = ti.Vector.var(2, ti.f32, N)
potential = ti.var(ti.f32, (), needs_grad=True)


@ti.kernel
def calc_potential():
    for i, j in ti.ndrange(N, N):
        disp = pos[i] - pos[j]
        potential[None] += 1 / disp.norm(1e-3)


@ti.kernel
def init():
    for i in pos:
        pos[i] = [ti.random(), ti.random()]


@ti.kernel
def advance():
    for i in pos:
        vel[i] += dt * pos.grad[i]
    for i in pos:
        pos[i] += dt * vel[i]


def substep():
    with ti.Tape(potential):
        calc_potential()
    advance()


init()
gui = ti.GUI('Autodiff gravity')
while gui.running and not gui.get_event(gui.ESCAPE):
    for i in range(16):
        substep()
    gui.circles(pos.to_numpy(), radius=3)
    gui.show()
