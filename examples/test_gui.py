import taichi as tc

vec = tc.Vector

gui = tc.core.GUI("Test GUI", tc.Vectori(512, 512))
canvas = gui.get_canvas()
canvas.rect(vec(0.3, 0.3), vec(0.6, 0.6)).radius(2).color(0xFFFFFF).close()

while True:
  gui.update()
