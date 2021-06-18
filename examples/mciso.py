import taichi as ti
import numpy as np
ti.init()

N = 128

_et = np.array(
    [
        [[-1, -1], [-1, -1]],  #
        [[0, 1], [-1, -1]],  #a
        [[0, 2], [-1, -1]],  #b
        [[1, 2], [-1, -1]],  #ab
        [[1, 3], [-1, -1]],  #c
        [[0, 3], [-1, -1]],  #ca
        [[2, 3], [0, 1]],  #cb
        [[2, 3], [-1, -1]],  #cab
        [[2, 3], [-1, -1]],  #d
        [[2, 3], [0, 1]],  #da
        [[0, 3], [-1, -1]],  #db
        [[1, 3], [-1, -1]],  #dab
        [[1, 2], [-1, -1]],  #dc
        [[0, 2], [-1, -1]],  #dca
        [[0, 1], [-1, -1]],  #dcb
        [[-1, -1], [-1, -1]],  #dcab
    ],
    np.int32)

m = ti.field(float, (N, N))  # field value of voxels

r = ti.Vector.field(2, float, (N**2, 2))  # generated edges
et = ti.Vector.field(2, int, _et.shape[:2])  # edge table (constant)
et.from_numpy(_et)


@ti.func
def gauss(x):
    return ti.exp(-6 * x**2)


@ti.kernel
def touch(mx: float, my: float):
    for i, j in m:
        p = ti.Vector([i, j]) / N
        a = gauss((p - 0.5).norm() / 0.25)
        p.x -= mx - 0.5 / N
        p.y -= my - 0.5 / N
        b = gauss(p.norm() / 0.25)
        m[i, j] = (a + b) * 3


@ti.func
def list_subscript(a,
                   i):  # magic method to subscript a list with dynamic index
    ret = sum(a) * 0
    for j in ti.static(range(len(a))):
        if i == j:
            ret = a[j]
    return ret


@ti.kernel
def march() -> int:
    r_n = 0

    for i, j in ti.ndrange(N - 1, N - 1):
        id = 0
        if m[i, j] > 1: id |= 1
        if m[i + 1, j] > 1: id |= 2
        if m[i, j + 1] > 1: id |= 4
        if m[i + 1, j + 1] > 1: id |= 8

        E = [ti.Vector(_) + .5 for _ in [(.5, 0), (0, .5), (1, .5), (.5, 1)]]

        for k in range(2):
            if et[id, k][0] == -1:
                break

            n = ti.atomic_add(r_n, 1)
            for l in ti.static(range(2)):
                e = et[id, k][l]
                r[n, l] = ti.Vector([i, j]) + list_subscript(E, e)

    return r_n


gui = ti.GUI('Marching rect')
while gui.running and not gui.get_event(gui.ESCAPE):
    touch(*gui.get_cursor_pos())
    ret_len = march()
    ret = r.to_numpy()[:ret_len] / N
    gui.set_image(ti.imresize(m, *gui.res))
    gui.lines(ret[:, 0], ret[:, 1], color=0xff66cc, radius=1.5)
    gui.show()
