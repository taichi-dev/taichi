import taichi as ti

x, y = 0.5, 0.5
delta = 0.01
radius = 8

gui = ti.GUI("Keyboard", res=(400, 400))

while gui.running:
    while gui.get_event(ti.GUI.PRESS, ti.GUI.MOTION):
        if gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
        elif gui.event.key == ti.GUI.RMB:
            x, y = gui.event.pos
        elif gui.event.key == ti.GUI.WHEEL:
            x, y = gui.event.pos
            dt = gui.event.delta
            # delta is 2-dim vector (dx, dy)
            # dt[0] denotes the horizontal direction, and dt[1] denotes the vertical direction
            if dt[1] > 0:
                radius += 10
            elif dt[1] < 0:
                radius = max(8, radius - 10)

    if gui.is_pressed(ti.GUI.LEFT, "a"):
        x -= delta
    if gui.is_pressed(ti.GUI.RIGHT, "d"):
        x += delta
    if gui.is_pressed(ti.GUI.UP, "w"):
        y += delta
    if gui.is_pressed(ti.GUI.DOWN, "s"):
        y -= delta
    if gui.is_pressed(ti.GUI.LMB):
        x, y = gui.get_cursor_pos()

    gui.text(f"({x:.3}, {y:.3})", (x, y))

    gui.circle((x, y), 0xFFFFFF, radius)
    gui.show()
