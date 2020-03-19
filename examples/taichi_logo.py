import taichi as ti

n = 512
x = ti.var(ti.f32, shape=[n, n])


@ti.func
def Vector2(x, y):
    return ti.Vector([x, y])


@ti.func
def inside(p, c, r):
    return (p - c).norm_sqr() <= r * r


@ti.func
def inside_taichi(p):
    p = Vector2(0.5, 0.5) + (p - Vector2(0.5, 0.5)) * 1.11
    ret = -1
    if not inside(p, Vector2(0.50, 0.50), 0.55):
        if ret == -1:
            ret = 0
    if not inside(p, Vector2(0.50, 0.50), 0.50):
        if ret == -1:
            ret = 1
    if inside(p, Vector2(0.50, 0.25), 0.09):
        if ret == -1:
            ret = 1
    if inside(p, Vector2(0.50, 0.75), 0.09):
        if ret == -1:
            ret = 0
    if inside(p, Vector2(0.50, 0.25), 0.25):
        if ret == -1:
            ret = 0
    if inside(p, Vector2(0.50, 0.75), 0.25):
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return ret


@ti.kernel
def paint():
    for i, j in ti.ndrange(n * 4, n * 4):
        ret = 1 - inside_taichi(Vector2(i / n / 4, j / n / 4))
        x[i // 4, j // 4] += ret / 16


paint()

gui = ti.GUI('Logo', (512, 512))
while True:
    gui.set_image(x.to_numpy())
    gui.show()
