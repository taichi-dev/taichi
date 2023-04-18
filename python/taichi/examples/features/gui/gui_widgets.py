import taichi as ti

gui = ti.GUI("GUI widgets")

radius = gui.slider("Radius", 1, 50, step=1)
xcoor = gui.label("X-coordinate")
okay = gui.button("OK")

xcoor.value = 0.5
radius.value = 10

while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == "a":
            xcoor.value -= 0.05
        elif e.key == "d":
            xcoor.value += 0.05
        elif e.key == "s":
            radius.value -= 1
        elif e.key == "w":
            radius.value += 1
        elif e.key == okay:
            print("OK clicked")

    gui.circle((xcoor.value, 0.5), radius=radius.value)
    gui.show()
