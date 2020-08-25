import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

RES = 1024
K = 2
R = 7
N = K**R

Broot = ti.root
B = ti.root
for r in range(R):
    B = B.bitmasked(ti.ij, (K, K))

qt = ti.field(ti.f32)
B.place(qt)

img = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))

print('The quad tree layout is:\n', qt.snode)


@ti.kernel
def action(p: ti.ext_arr()):
    a = ti.cast(p[0] * N, ti.i32)
    b = ti.cast(p[1] * N, ti.i32)
    qt[a, b] = 1


@ti.func
def draw_rect(b: ti.template(), i, j, s, k, dx, dy):
    x = i // s
    y = j // s
    a = 0
    if dx and i % k == 0 or dy and j % k == 0:
        a += ti.is_active(b, [x, y])
        a += ti.is_active(b, [x - dx, y - dy])
    return a


@ti.kernel
def paint():
    for i, j in img:
        for k in ti.static(range(3)):
            img[i, j][k] *= 0.85
    for i, j in img:
        s = RES // N
        for r in ti.static(range(R)):
            k = RES // K**(R - r)
            ia = draw_rect(qt.parent(r + 1), i, j, s, k, 1, 0)
            ja = draw_rect(qt.parent(r + 1), i, j, s, k, 0, 1)
            img[i, j][0] += (ia + ja) * ((R - r) / R)**2


def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)


gui = ti.GUI('Quadtree', (RES, RES))
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    Broot.deactivate_all()
    pos = gui.get_cursor_pos()
    action(vec2_npf32(pos))
    paint()
    gui.set_image(img.to_numpy())
    gui.show()
