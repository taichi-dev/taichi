import taichi as ti
from math import tau
ti.init()

N = 512
img = ti.var(dt=ti.f32, shape=(N, N))
light_pos = ti.Vector(2, dt=ti.f32, shape=())

@ti.func
def vres(distance, emission, refl_rate, refl_color):
    return ti.Vector([distance, emission, refl_rate, refl_color])

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
def reflect(r, n):
    return r - 2 * r.dot(n) * n

@ti.func
def subtract(a, b):
    if a[0] < -b[0]:
        a = b
        a[0] = -a[0]
    return a

@ti.func
def sdf(p):
    #       EMI; RFL, CLR
    d1 = vres((p - light_pos + vec2(0.05, 0.0)).norm() - 0.1,
            1.0, 0.0, 0.0)
    d2 = vres((p - light_pos - vec2(0.05, 0.0)).norm() - 0.1,
            1.0, 0.0, 0.0)
    d3 = vres(p[1] - 0.6,
            0.0, 1.0, 1.0)
    d4 = vres((p - vec2(0.5, 0.6)).norm() - 0.3,
            0.0, 1.0, 1.0)
    return union(subtract(d1, d2), subtract(d3, d4))

@ti.func
def gradient(p):  # ASK(yuanming-hu): do we have sdf.grad?
    ax = sdf(p + vec2(+1e-4, 0))[0]
    bx = sdf(p + vec2(-1e-4, 0))[0]
    ay = sdf(p + vec2(0, +1e-4))[0]
    by = sdf(p + vec2(0, -1e-4))[0]
    return ti.Vector.normalized(vec2(ax - bx, ay - by))

@ti.func
def sample(p):
    a = ti.random() * tau
    d = ti.Vector([ti.cos(a), ti.sin(a)])
    ret = 0.0
    fac = 1.0
    depth = 0
    steps = 0
    while depth < 5 and steps < 1e4:
        steps += 1
        f = sdf(p)
        f[0] = ti.max(f[0], 0.0)
        p += d * f[0]
        if f[0] < 1e-4:
            ret += fac * f[1]
            if f[2] != 0.0:  # REL: #832
                if ti.random() < f[2]:
                    depth += 1
                    fac *= f[3]
                    n = gradient(p)
                    d = reflect(d, n)
                    p += n * 1e-3
                else:
                    break
            else:
                break
        elif f[0] > 1e1:
            break
        else:
            pass
    return ret

@ti.kernel
def render():
    for i, j in img:
        o = ti.Vector([i / N, j / N])
        img[i, j] += sample(o)

gui = ti.GUI('SDF 2D')
frame = 1
light_pos[None] = [0.85, 0.75]
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
