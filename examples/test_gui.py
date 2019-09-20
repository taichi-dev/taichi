import taichi as tc

vec = tc.Vector

gui = tc.core.GUI("Test GUI", tc.Vectori(512, 512))
canvas = gui.get_canvas()
canvas.clear(0xFFFFFF)
canvas.rect(vec(0.3, 0.3), vec(0.6, 0.6)).radius(2).color(0x000000).close()
canvas.path(vec(0.3, 0.3), vec(0.6, 0.6)).radius(2).color(0x000000).close()
canvas.circle(vec(0.5, 0.5)).radius(10).color(0xFF0000)

while True:
  gui.update()
