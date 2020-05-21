import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

n = 100
celsiz = 4
res = n * celsiz
x = ti.var(ti.i32, shape=(n, n))
c = ti.var(ti.i32, shape=(n, n))
img = ti.var(ti.i32, shape=(res, res))


@ti.func
def count(i, j):
    return (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1] +
            x[i - 1, j - 1] + x[i + 1, j - 1] + x[i - 1, j + 1] +
            x[i + 1, j + 1])


@ti.kernel
def run():
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        c[i, j] = count(i, j)
    for i, j in ti.ndrange((1, n - 1), (1, n - 1)):
        if x[i, j] == 0:
            if c[i, j] == 3:
                x[i, j] = 1
        elif c[i, j] != 2 and c[i, j] != 3:
            x[i, j] = 0


@ti.kernel
def init():
    for i, j in x:
        if ti.random() > 0.8:
            x[i, j] = 1
        else:
            x[i, j] = 0


@ti.kernel
def render():
    for i, j in x:
        c = 0
        if x[i, j] != 0: c = 255
        for u, v in ti.ndrange(celsiz, celsiz):
            img[i * celsiz + u, j * celsiz + v] = c


init()
gui = ti.GUI('Game of Life', (res, res))
print('Press the spacebar to run.')

render()
while True:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.SPACE:
            run()
            render()
        elif gui.event.key == ti.GUI.ESCAPE:
            exit()
    gui.set_image(img.to_numpy().astype(np.uint8))
    gui.show()
