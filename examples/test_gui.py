import taichi as tc

gui = tc.core.GUI("Test GUI", tc.Vectori(512, 512))

while True:
  gui.update()
