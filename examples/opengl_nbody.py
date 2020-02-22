import taichi as ti

ti.init(arch=ti.opengl)

N = 8000
pos = ti.var(dt=ti.f32, shape=(N, 2))
vel = ti.var(dt=ti.f32, shape=(N, 2))
bounce = 0.8

@ti.kernel
def initialize():
  for i in range(N):
    pos[i, 0] = ti.random() * 0.5 - 0.25 + 0.5
    pos[i, 1] = ti.random() * 0.5 - 0.25 + 0.5
    vel[i, 0] = ti.random() * 2 - 1
    vel[i, 1] = ti.random() * 2 - 1

@ti.kernel
def advance(dt: ti.f32):
  for i in range(N):
    if pos[i, 0] < 0 and vel[i, 0] < 0 or pos[i, 0] > 1 and vel[i, 0] > 0:
      vel[i, 0] = -bounce * vel[i, 0]
    if pos[i, 1] < 0 and vel[i, 1] < 0 or pos[i, 1] > 1 and vel[i, 1] > 0:
      vel[i, 1] = -bounce * vel[i, 1]

    px = pos[i, 0] - 0.5
    py = pos[i, 1] - 0.5
    pas = -80.0 * ti.sqrt(px ** 2 + py ** 2)
    a_x = px * pas
    a_y = py * pas

    for j in range(N):
      if i != j:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d2 = dx ** 2 + dy ** 2
        das = 233.0 * 0.001 ** 2 / (0.001 ** 2 + d2)
        a_x = a_x + dx * das
        a_y = a_y + dy * das

    pos[i, 0] = pos[i, 0] + vel[i, 0] * dt
    pos[i, 1] = pos[i, 1] + vel[i, 1] * dt

    vel[i, 0] = vel[i, 0] + a_x * dt
    vel[i, 1] = vel[i, 1] + a_y * dt

gui = ti.GUI("n-body", res=(400, 400))

initialize()
while not gui.has_key_event() or gui.get_key_event().key == ti.GUI.MOTION:
  _pos = pos.to_numpy()
  gui.circles(_pos, radius=1, color=0x66ccff)
  gui.show()
  advance(0.005)
