import taichi as ti

N = 8
dt = 1e-5

x = ti.Vector.field(2, float, N, needs_grad=True)  # position of particles
v = ti.Vector.field(2, float, N)  # velocity of particles
U = ti.field(float, (), needs_grad=True)  # potential energy


@ti.kernel
def compute_U():
    for i, j in ti.ndrange(N, N):
        r = x[i] - x[j]
        # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
        # This is to prevent 1/0 error which can cause wrong derivative
        U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|


@ti.kernel
def advance():
    for i in x:
        v[i] += dt * -x.grad[i]  # dv/dt = -dU/dx
    for i in x:
        x[i] += dt * v[i]  # dx/dt = v


def substep():
    with ti.Tape(U):
        # every kernel invocation within this indent scope
        # will also be accounted into the partial derivate of U
        # with corresponding input variables like x.
        compute_U()  # will also computes dU/dx and save in x.grad
    advance()


@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random(), ti.random()]


init()
gui = ti.GUI('Autodiff gravity')
while gui.running:
    for i in range(50):
        substep()
    print('U = ', U[None])
    gui.circles(x.to_numpy(), radius=3)
    gui.show()
