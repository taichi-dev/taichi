import taichi as ti

ti.init(arch=ti.gpu)

N = 8000
pos = ti.Vector(2, dt=ti.f32, shape=N)
vel = ti.Vector(2, dt=ti.f32, shape=N)
bounce = 0.8


@ti.kernel
def initialize():
    for i in range(N):
        for k in ti.static(range(2)):
            pos[i][k] = ti.random() * 0.5 - 0.25 + 0.5
            vel[i][k] = ti.random() * 2 - 1


@ti.kernel
def advance(dt: ti.f32):
    for i in range(N):
        for k in ti.static(range(2)):
            if pos[i][k] < 0 and vel[i][k] < 0 or pos[i][k] > 1 and vel[i][
                    k] > 0:
                vel[i][k] = -bounce * vel[i][k]

        p = pos[i] - ti.Vector([0.5, 0.5])
        pas = -80.0 * p.norm()
        a = p * pas

        for j in range(N):
            if i != j:
                d = pos[i] - pos[j]
                d2 = d.norm_sqr()
                das = 233.0 * 0.001**2 / (0.001**2 + d2)
                a = a + d * das

        pos[i] = pos[i] + vel[i] * dt
        vel[i] = vel[i] + a * dt


gui = ti.GUI("n-body", res=(400, 400))

initialize()
while not gui.get_event(ti.GUI.ESCAPE):
    _pos = pos.to_numpy()
    gui.circles(_pos, radius=1, color=0x66ccff)
    gui.show()
    advance(0.005)
