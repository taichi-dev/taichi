import sys

import numpy as np

import taichi as ti

try:
    import taichi_glsl as tl
    import taichi_three as t3
except ImportError:
    print('This example needs the extension library Taichi THREE to work.'
          'Please run `pip install taichi_three==0.0.5` to install it.')
    sys.exit(1)
if t3.__version__ != (0, 0, 5):
    print('WARNING: Taichi THREE v0.0.5 expected, got v' +
          '.'.join(map(str, t3.__version__)))

ti.init(arch=ti.gpu)

### Parameters

N = 128
NN = N, N
W = 1
L = W / N
gravity = 0.5
stiffness = 1600
damping = 2
steps = 30
dt = 5e-4

### Physics

x = ti.Vector.field(3, float, NN)
v = ti.Vector.field(3, float, NN)


@ti.kernel
def init():
    for i in ti.grouped(x):
        x[i] = tl.vec((i + 0.5) * L - 0.5, 0.8).xzy


links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [tl.vec(*_) for _ in links]


@ti.kernel
def substep():
    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[tl.clamp(i + d, 0, tl.vec(*NN) - 1)] - x[i]
            length = L * float(d).norm()
            acc += disp * (disp.norm() - length) / length**2
        v[i] += stiffness * acc * dt
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
        v[i] = tl.ballBoundReflect(x[i], v[i], tl.vec(+0.0, +0.2, -0.0), 0.4,
                                   6)
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        x[i] += dt * v[i]


### Rendering GUI

scene = t3.Scene()
model = t3.Model(faces_n=(N - 1)**2 * 4,
                 pos_n=N**2,
                 tex_n=N**2,
                 nrm_n=N**2 * 2)
#model.load_texture(ti.imread('/path/to/your_texture.jpg'))
scene.add_model(model)
camera = t3.Camera(fov=24, pos=[0, 1.1, -1.5], target=[0, 0.25, 0])
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, 1.8])
scene.add_light(light)


@ti.kernel
def init_display():
    for i_ in ti.grouped(ti.ndrange(N - 1, N - 1)):
        i = i_
        a = i.dot(tl.vec(N, 1))
        i.x += 1
        b = i.dot(tl.vec(N, 1))
        i.y += 1
        c = i.dot(tl.vec(N, 1))
        i.x -= 1
        d = i.dot(tl.vec(N, 1))
        i.y -= 1
        for p in ti.static(range(3)):
            for q in ti.static(range(3)):
                model.faces[a * 4 + 0][p, q] = [a, c, b][p]
                model.faces[a * 4 + 1][p, q] = [a, d, c][p]
        for p in ti.static(range(3)):
            for q in ti.static(range(2)):
                model.faces[a * 4 + 2][p, q] = [a, b, c][p]
                model.faces[a * 4 + 3][p, q] = [a, c, d][p]
        a += N**2
        b += N**2
        c += N**2
        d += N**2
        for p in ti.static(range(3)):
            model.faces[a * 4 + 2][p, 2] = [a, b, c][p]
            model.faces[a * 4 + 3][p, 2] = [a, c, d][p]
    for i in ti.grouped(x):
        j = i.dot(tl.vec(N, 1))
        model.tex[j] = tl.D._x + i.xY / N


@ti.kernel
def update_display():
    for i in ti.grouped(x):
        j = i.dot(tl.vec(N, 1))
        model.pos[j] = x[i]

        xa = x[tl.clamp(i + tl.D.x_, 0, tl.vec(*NN) - 1)]
        xb = x[tl.clamp(i + tl.D.X_, 0, tl.vec(*NN) - 1)]
        ya = x[tl.clamp(i + tl.D._x, 0, tl.vec(*NN) - 1)]
        yb = x[tl.clamp(i + tl.D._X, 0, tl.vec(*NN) - 1)]
        normal = (ya - yb).cross(xa - xb).normalized()
        model.nrm[j] = normal
        model.nrm[N**2 + j] = -normal


init()
init_display()

with ti.GUI('Mass Spring') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        for i in range(steps):
            substep()
        update_display()

        camera.from_mouse(gui)

        scene.render()
        gui.set_image(camera.img)
        gui.show()
