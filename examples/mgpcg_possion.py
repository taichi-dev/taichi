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

@ti.func
def sample(x: ti.template(), u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(n - 1, I))
    return x[I]

@ti.kernel
def compute_loss() -> float:
    ret = 0.0
    for i, j in x:
        xc = sample(x, i, j)
        xl = sample(x, i - 1, j)
        xr = sample(x, i + 1, j)
        xb = sample(x, i, j - 1)
        xt = sample(x, i, j + 1)
        div = sample(b, i, j)
        ret += xl + xr + xb + xt - 4 * xc + div
    return ret

gui = ti.GUI('MGPCG Possion Solver', (n, n))
while gui.running and not gui.get_event(gui.ESCAPE):
    touch_at(*gui.get_cursor_pos())
    mgpcg.init(b, 1)
    for i, rTr in mgpcg.solve(steps):
        pass#print(f'iter {i}, residual={rTr}')
    mgpcg.get_result(x)
    #print('loss', compute_loss())
    print('b', np.max(b.to_numpy()))
    print('x', np.max(x.to_numpy()))
    gui.set_image(x)
    gui.show()
