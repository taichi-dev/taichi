import taichi as ti

x, y = 0.5, 0.5
delta = 0.01

gui = ti.GUI("Keyboard", res=(400, 400))

while True:
  while gui.has_key_event():
    e = gui.get_key_event()
    if e.type == ti.GUI.RELEASE:
      continue
    if e.key == ti.GUI.ESCAPE:
      exit()
    elif e.key == ti.GUI.RMB:
      x, y = e.pos[0], e.pos[1]

  if gui.is_pressed(ti.GUI.LEFT, 'a'):
    x -= delta
  if gui.is_pressed(ti.GUI.RIGHT, 'd'):
    x += delta
  if gui.is_pressed(ti.GUI.UP, 'w'):
    y += delta
  if gui.is_pressed(ti.GUI.DOWN, 's'):
    y -= delta
  if gui.is_pressed(ti.GUI.LMB):
    pos = gui.get_cursor_pos()
    x, y = pos[0], pos[1]

  gui.circle((x, y), 0xffffff, 8)
  gui.show()
