import math

import taichi as ti

ti.init(arch=ti.cuda)

dim = 3
N = 1024 * 8
dt = 2e-4
steps = 7
sun = ti.Vector([0.5, 0.5] if dim == 2 else [0.5, 0.5, 0.0])
gravity = 0.5
pressure = 0.3
tail_paticle_scale = 0.4
color_init = 0.3
color_decay = 1.6
vel_init = 0.07
res = 640

inv_m = ti.field(ti.f32)
color = ti.field(ti.f32)
x = ti.Vector.field(dim, ti.f32)
v = ti.Vector.field(dim, ti.f32)
ti.root.bitmasked(ti.i, N).place(x, v, inv_m, color)
count = ti.field(ti.i32, ())
img = ti.field(ti.f32, (res, res))


@ti.func
def rand_unit_2d():
    a = ti.random() * 2 * math.pi
    return ti.Vector([ti.cos(a), ti.sin(a)])


@ti.func
def rand_unit_3d():
    u = rand_unit_2d()
    s = ti.random() * 2 - 1
    c = ti.sqrt(1 - s**2)
    return ti.Vector([c * u[0], c * u[1], s])


@ti.kernel
def substep():
    ti.no_activate(x)
    for i in x:
        r = x[i] - sun
        r_sq_inverse = r / r.norm(1e-3) ** 3
        acceleration = (pressure * inv_m[i] - gravity) * r_sq_inverse
        v[i] += acceleration * dt
        x[i] += v[i] * dt
        color[i] *= ti.exp(-dt * color_decay)

        if not all(-0.1 <= x[i] <= 1.1):
            ti.deactivate(x.snode.parent(), [i])


@ti.kernel
def generate():
    r = x[0] - sun
    n_tail_paticles = int(tail_paticle_scale / r.norm(1e-3) ** 2)
    for _ in range(n_tail_paticles):
        r = x[0]
        if ti.static(dim == 3):
            r = rand_unit_3d()
        else:
            r = rand_unit_2d()
        xi = ti.atomic_add(count[None], 1) % (N - 1) + 1
        x[xi] = x[0]
        v[xi] = r * vel_init + v[0]
        inv_m[xi] = 0.5 + ti.random()
        color[xi] = color_init


@ti.kernel
def render():
    for p in ti.grouped(img):
        img[p] = 1e-6 / (p / res - ti.Vector([sun.x, sun.y])).norm(1e-4) ** 3
    for i in x:
        p = int(ti.Vector([x[i].x, x[i].y]) * res)
        if 0 <= p[0] < res and 0 <= p[1] < res:
            img[p] += color[i]


def main():
    inv_m[0] = 0
    x[0].x = +0.5
    x[0].y = -0.01
    v[0].x = +0.6
    v[0].y = +0.4
    color[0] = 1

    gui = ti.GUI("Comet", res)
    while gui.running:
        gui.running = not gui.get_event(gui.ESCAPE)
        generate()
        for s in range(steps):
            substep()
        render()
        gui.set_image(img)
        gui.show()


if __name__ == "__main__":
    main()
