import taichi as ti

x, y = 0.5, 0.5
delta = 0.01

gui = ti.GUI("Keyboard", res=(400, 400))

while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
        elif gui.event.key == ti.GUI.RMB:
            x, y = gui.event.pos

    if gui.is_pressed(ti.GUI.LEFT, 'a'):
        x -= delta
    if gui.is_pressed(ti.GUI.RIGHT, 'd'):
        x += delta
    if gui.is_pressed(ti.GUI.UP, 'w'):
        y += delta
    if gui.is_pressed(ti.GUI.DOWN, 's'):
        y -= delta
    if gui.is_pressed(ti.GUI.LMB):
        x, y = gui.get_cursor_pos()

    gui.text(f'({x:.3}, {y:.3})', (x, y))

    gui.circle((x, y), 0xffffff, 8)
    gui.show()
