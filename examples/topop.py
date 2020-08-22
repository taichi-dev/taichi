import taichi as ti
import numpy as np

ti.init(ti.cpu)

N = 9
NN = N, N
dt = 1e-2
stiffness = 1
gravity = 1

m = ti.field(float, NN, needs_grad=True)
h = ti.field(float, NN, needs_grad=True)
F = ti.field(float, NN, needs_grad=True)
U = ti.field(float, (), needs_grad=True)


links = list(map(ti.Vector, [(-1, 0), (1, 0), (0, -1), (0, 1)]))


@ti.kernel
def advance_deformation():
    for I in ti.grouped(h):
        neigh = h[I] * 0
        for dI in ti.static(links):
            neigh += h[I + dI] - h[I]
        h[I] += stiffness / len(links) * neigh * dt
        h[I] += gravity * F[I] * dt

@ti.kernel
def compute_deformation():
    for I in ti.grouped(m):
        if m[I] != 0:
            h[I] /= m[I]
        U[None] += h[I]**2

@ti.kernel
def compute_mass_penalty():
    for _ in range(1):  # see https://github.com/taichi-dev/taichi/issues/1746
        mass = 0.0
        for I in ti.static(ti.grouped(ti.ndrange(N, N))):
            mass += abs(m[I]) - 1
        U[None] += 3e-2 * (max(mass, 0) / N**2)**2

@ti.kernel
def optimize_via_gradient():
    for I in ti.grouped(m):
        m[I] = max(0, m[I] - dt * m.grad[I])

@ti.kernel
def set_boundary_condition():
    F[N // 2 + 2, N // 2 + 2] = 1
    F[N // 2 - 2, N // 2 - 2] = 1
    F[N // 2, N // 2] = -1


m.fill(1)
set_boundary_condition()
gui1 = ti.GUI('Height field')
gui2 = ti.GUI('Mass field')
while gui1.running and gui2.running:
    gui1.running = not gui1.get_event(ti.GUI.ESCAPE)
    gui2.running = not gui2.get_event(ti.GUI.ESCAPE)
    for s in range(10):
        advance_deformation()
    with ti.Tape(U):
        compute_deformation()
        compute_mass_penalty()
    print(U[None])
    _ = h.to_numpy() * 8
    gui1.point_field(np.maximum(0, _), color=0xffcc66)
    gui1.point_field(np.maximum(0, -_), color=0xcc66ff)
    _ = m.to_numpy() * 6
    gui2.point_field(np.abs(_), color=0x66ffcc)
    gui1.show()
    gui2.show()
