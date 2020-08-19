import taichi as ti
import numpy as np
from mgpcg_advanced import MGPCG
ti.init(arch=ti.cuda)

n = 256
steps = 32

b = ti.field(float, (n, n))
x = ti.field(float, (n, n))

mgpcg = MGPCG(dim=2, N=n)

@ti.kernel
def touch_at(mx: float, my: float):
    for I in ti.grouped(ti.ndrange(n, n)):
        d = I / n - ti.Vector([mx, my])
        b[I] = 1e-2 * ti.exp(-1e3 * d.norm_sqr())

gui = ti.GUI('MGPCG Possion Solver', (n, n))
while gui.running and not gui.get_event(gui.ESCAPE):
    touch_at(*gui.get_cursor_pos())
    mgpcg.init(b, 1)
    for i, rTr in mgpcg.solve(steps):
        print(f'iter {i}, residual={rTr}')
    mgpcg.get_result(x)
    gui.set_image(x)
    gui.show()
