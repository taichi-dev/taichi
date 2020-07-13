# Game of Life written in 81 lines of Taichi
# In memory of John Horton Conway (1937 - 2020)

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

n = 64
cell_size = 8
img_size = n * cell_size
alive = ti.var(ti.i32, shape=(n, n))  # alive = 1, dead = 0
count = ti.var(ti.i32, shape=(n, n))  # count of neighbours
img = ti.var(ti.f32, shape=(img_size, img_size))  # image to be displayed


@ti.func
def get_count(i, j):
    return (alive[i - 1, j] + alive[i + 1, j] + alive[i, j - 1] + alive[i, j + 1] +
            alive[i - 1, j - 1] + alive[i + 1, j - 1] + alive[i - 1, j + 1] +
            alive[i + 1, j + 1])


# https://www.conwaylife.com/wiki/Cellular_automaton#Rules

@ti.func
def rule_3_23(a, c):
    if a == 0 and c == 3:
        a = 1
    elif a == 1 and c != 2 and c != 3:
        a = 0
    return a

@ti.func
def rule_2_0(a, c):
    if a == 0 and c == 2:
        a = 1
    elif a == 1 and c != 0:
        a = 0
    return a

@ti.func
def rule_3_1234(a, c):
    if a == 0 and c == 3:
        a = 1
    elif a == 1 and (c == 0 or c >= 5):
        a = 0
    return a

@ti.func
def rule_3_125(a, c):
    if a == 0 and c == 3:
        a = 1
    elif a == 1 and c != 1 and c != 2 and c != 5:
        a = 0
    return a

@ti.func
def rule_1357_1357(a, c):
    tmp = c != 1 and c != 3 and c != 5 and c != 7
    if a == 0 and not tmp:
        a = 1
    elif a == 1 and tmp:
        a = 0
    return a


@ti.kernel
def run():
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        count[i, j] = get_count(i, j)

    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        alive[i, j] = rule_3_23(alive[i, j], count[i, j])


@ti.kernel
def init():
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if ti.random() > 0.8:
            alive[i, j] = 1
        else:
            alive[i, j] = 0


@ti.kernel
def render():
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        for u, v in ti.static(ti.ndrange(cell_size, cell_size)):
            img[i * cell_size + u, j * cell_size + v] = alive[i, j]


gui = ti.GUI('Game of Life', (img_size, img_size))

print('[Hint] Press `r` to reset')
print('[Hint] Press SPACE to pause')
print('[Hint] Click LMB, RMB and drag to add alive / dead cells')

init()
paused = False
while gui.running:
    for e in gui.get_events(gui.PRESS, gui.MOTION):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.SPACE:
            paused = not paused
        elif e.key == 'r':
            alive.fill(0)

    if gui.is_pressed(gui.LMB, gui.RMB):
        mx, my = gui.get_cursor_pos()
        alive[int(mx * n), int(my * n)] = gui.is_pressed(gui.LMB)
        paused = True

    if not paused and gui.frame % 4 == 0:
        run()

    render()

    gui.set_image(img)
    gui.show()
