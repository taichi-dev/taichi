import taichi as ti
import sys

x, y = 0.5, 0.5
delta = 0.01

gui = ti.GUI("Keyboard", res=(400, 400))

while True:
  while gui.has_key_event():
    key, type = gui.get_key_event()
    if type == ti.GUI.RELEASE:
      continue
    if key == ti.GUI.ESCAPE:
      sys.exit()

  if gui.is_pressed(ti.GUI.LEFT, 'a'):
    x -= delta
  if gui.is_pressed(ti.GUI.RIGHT, 'd'):
    x += delta
  if gui.is_pressed(ti.GUI.UP, 'w'):
    y += delta
  if gui.is_pressed(ti.GUI.DOWN, 's'):
    y -= delta

  gui.circle((x, y), 0x66ccff, 3)
  gui.show()
