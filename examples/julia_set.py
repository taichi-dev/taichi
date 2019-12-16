import taichi as ti

n = 512
pixels = ti.Vector(3, dt=ti.f32, shape=(n, n))
ti.cfg.print_ir = True
# ti.get_runtime().print_preprocessed = True

@ti.func
def complex_sqr(z):
  return ti.Vector([z[0] * z[0] - z[1] * z[1], z[1] * z[0] * 2]) # z^2

@ti.kernel
def paint():
  for i, j in pixels:
    c = ti.Vector([-0.8, 0.156])
    z = ti.Vector([float(i) / n, float(j) / n])

    iterations = 0
    while z.norm() < 20 and iterations < 50:
      z = complex_sqr(z) + c
      iterations += 1

    pixels[i, j] = [iterations * 0.05, 0, 0]

gui = ti.core.GUI("Julia Set", ti.veci(n, n))
canvas = gui.get_canvas()

while True:
  paint()
  gui.set_image(pixels)
  gui.update()
