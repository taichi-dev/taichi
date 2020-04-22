import taichi as ti
from math import tau
from renderer_utils import reflect, refract
ti.init()

N = 512
img = ti.var(dt=ti.f32, shape=(N, N))
light_pos = ti.Vector(2, dt=ti.f32, shape=())

@ti.func
def vres(distance, emission, reflection, refraction):
    return ti.Vector([distance, emission, reflection, refraction])

@ti.func
def vec2(x, y):
    return ti.Vector([x, y])

@ti.func
def vec3(x, y, z):
    return ti.Vector([x, y, z])

@ti.func
def union(a, b):
    if a[0] > b[0]:
        a = b
    return a

@ti.func
def intersect(a, b):
    if a[0] < b[0]:
        a = b
    return a

@ti.func
def subtract(a, b):
    if a[0] < -b[0]:
        a = b
        a[0] = -a[0]
    return a

@ti.func
def sdf_moon(p):
    #       EMI, RFL, RFR
    d1 = vres((p - light_pos + vec2(0.05, 0.0)).norm() - 0.1,
            1.0, 0.0, 0.0)
    d2 = vres((p - light_pos - vec2(0.05, 0.0)).norm() - 0.1,
            1.0, 0.0, 0.0)
    d3 = vres(p[1] - 0.6,
            0.0, 1.0, 0.0)
    d4 = vres((p - vec2(0.5, 0.6)).norm() - 0.3,
            0.0, 1.0, 0.0)
    return union(subtract(d1, d2), subtract(d3, d4))

@ti.func
def sdf_lens(p):
    #       EMI, RFL, RFR
    d1 = vres((p - vec2(0.5, 0.4)).norm() - 0.15,
            0.0, 0.0, 1.0)
    d2 = vres((p - vec2(0.5, 0.6)).norm() - 0.15,
            0.0, 0.0, 1.0)
    d3 = vres((p - light_pos).norm() - 0.02,
            4.0, 0.0, 0.0)
    return union(intersect(d1, d2), d3)

sdf = sdf_lens

@ti.func
def gradient(p):  # ASK(yuanming-hu): do we have sdf.grad?
    ax = sdf(p + vec2(+1e-4, 0))[0]
    bx = sdf(p + vec2(-1e-4, 0))[0]
    ay = sdf(p + vec2(0, +1e-4))[0]
    by = sdf(p + vec2(0, -1e-4))[0]
    return ti.Vector.normalized(vec2(ax - bx, ay - by))

@ti.func
def random_in(n):
    ret = 0
    if n > 0:
        ret = ti.random() < n
    return ret

@ti.func
def sample(p):
    a = ti.random() * tau
    d = vec2(ti.cos(a), ti.sin(a))
    ret = 0.0
    depth = 0
    steps = 0
    sign = 1.0
    f = sdf(p)
    if f[0] < 0:
        sign = -1.0
    while depth < 5 and steps < 1e3:
        steps += 1
        f = sdf(p)
        p += d * f[0]
        if sign * f[0] < 1e-6:
            ret += f[1]  # EMI
            if random_in(f[2]):  # RFL
                depth += 1
                n = gradient(p)
                d = reflect(d, n)
                p += n * 1e-3
            elif random_in(f[3]):  # RFR
                depth += 1
                n = gradient(p) * sign
                has, d = refract(d, n, 1 / 1.33)
                if not has:
                    break
                p += n * 1e-3
            else:
                break
        elif abs(f[0]) > 1e1:
            break
        if f[0] < 0:
            sign = -1.0
        else:
            sign = 1.0
    return ret

@ti.kernel
def render():
    for i, j in img:
        o = ti.Vector([i / N, j / N])
        img[i, j] += sample(o)

gui = ti.GUI('SDF 2D')
frame = 1
light_pos[None] = [0.5, 0.85]
while True:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.LMB:
            light_pos[None] = [gui.event.pos[0], gui.event.pos[1]]
            frame = 1
            img.fill(0)
        elif gui.event.key == ti.GUI.ESCAPE:
            exit()
    render()
    gui.set_image(img.to_numpy() / frame)
    gui.show()
    frame += 1
