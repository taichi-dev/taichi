import taichi as ti
import numpy as np

ti.init(arch=ti.x64)

RES = 512
K = 2
R = 5
N = K ** R

b0 = ti.root
b1 = b0.bitmasked(ti.ij, (K, K))
b2 = b1.bitmasked(ti.ij, (K, K))
b3 = b2.bitmasked(ti.ij, (K, K))
b4 = b3.bitmasked(ti.ij, (K, K))
b5 = b4.bitmasked(ti.ij, (K, K))

qt = ti.var(ti.f32)
b5.place(qt)

img = ti.Vector(3, dt=ti.f32, shape=(RES, RES))

@ti.kernel
def action(p: ti.ext_arr()):
    a = ti.cast(p[0] * N, ti.i32)
    b = ti.cast(p[1] * N, ti.i32)
    qt[a, b] = 1

@ti.func
def actune(b, i, j, s, k, dx, dy):
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
        k = RES // K ** 1
        ia = actune(b1, i, j, s, k, 1, 0)
        ja = actune(b1, i, j, s, k, 0, 1)
        img[i, j][0] += (ia + ja) * 0.02
        k = RES // K ** 2
        ia = actune(b2, i, j, s, k, 1, 0)
        ja = actune(b2, i, j, s, k, 0, 1)
        img[i, j][0] += (ia + ja) * 0.04
        k = RES // K ** 3
        ia = actune(b3, i, j, s, k, 1, 0)
        ja = actune(b3, i, j, s, k, 0, 1)
        img[i, j][0] += (ia + ja) * 0.11
        k = RES // K ** 4
        ia = actune(b4, i, j, s, k, 1, 0)
        ja = actune(b4, i, j, s, k, 0, 1)
        img[i, j][0] += (ia + ja) * 0.15
        k = RES // K ** 5
        ia = actune(b5, i, j, s, k, 1, 0)
        ja = actune(b5, i, j, s, k, 0, 1)
        img[i, j][0] += (ia + ja) * 0.80

def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)

gui = ti.GUI('Quadtree', (RES, RES))
while not gui.get_event(ti.GUI.PRESS):
    b1.deactivate_all()
    b2.deactivate_all()
    b3.deactivate_all()
    b4.deactivate_all()
    pos = gui.get_cursor_pos()
    action(vec2_npf32(pos))
    paint()
    gui.set_image(img.to_numpy(as_vector=True))
    gui.show()
