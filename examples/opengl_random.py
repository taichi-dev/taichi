import taichi as ti

ti.init(arch=ti.opengl)

n = 640
pixels = ti.var(dt=ti.f32, shape=(n, n))

@ti.kernel
def paint():
  for i, j in pixels:
    pixels[i, j] = ti.random()

gui = ti.GUI("Noise", res=(n, n))

for i in range(1000000):
  paint()
  gui.set_image(pixels)
  gui.show()
  if gui.has_key_event() and gui.get_key_event().key != ti.GUI.MOTION:
    break
