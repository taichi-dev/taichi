import taichi as ti
import taichi_glsl as tl
import taichi_three as t3
import numpy as np
import math
ti.init(ti.gpu)

dt = 8e-3
N = 128
NN = N, N
W = 1
L = W / N
beta = 0.5
beta_dt = beta * dt
alpha_dt = (1 - beta) * dt
gravity = 0.4
stiff = 20
damp = 2.6

x = ti.Vector(3, ti.f32, NN)
v = ti.Vector(3, ti.f32, NN)
b = ti.Vector(3, ti.f32, NN)
F = ti.Vector(3, ti.f32, NN)


@ti.kernel
def init():
    for i in ti.grouped(x):
        x[i] = tl.vec((i + 0.5) * L - 0.5, 0.8).xzy


@ti.kernel
def e_accel():
    Acc(v, x, dt)


@ti.kernel
def e_update():
    for i in ti.grouped(x):
        v[i] *= math.exp(dt * -damp)
        x[i] += dt * v[i]


def explicit():
    '''
    v' = v + Mdt @ v
    x' = x + v'dt
    '''
    e_accel()
    collide(x, v)
    e_update()


#############################################
'''
v' = v + Mdt @ v'
(I - Mdt) @ v' = v
'''
'''
v' = v + Mdt @ [beta v' + alpha v]
(I - beta Mdt) @ v' = (I + alpha Mdt) @ v
'''

links = [tl.vec(*_) for _ in [(-1, 0), (1, 0), (0, -1), (0, 1)]]


@ti.func
def Acc(v: ti.template(), x: ti.template(), dt):
    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[tl.clamp(i + d, 0, tl.vec(*NN) - 1)] - x[i]
            dis = disp.norm()
            #disv = v[tl.clamp(i + d, 0, tl.vec(*NN) - 1)] - v[i]
            #acc += (damp / stiff) * disv * disv.dot(disp) / (dis * disv.norm_sqr())
            acc += disp * (dis - L) / L**2
        v[i] += stiff * acc * dt
        v[i] *= ti.exp(-damp * dt)


@ti.kernel
def prepare():
    if ti.static(beta != 1):
        for i in ti.grouped(x):
            x[i] += v[i] * alpha_dt
        Acc(v, x, alpha_dt)
    for i in ti.grouped(x):
        b[i] = x[i]
        x[i] += v[i] * beta_dt


@ti.kernel
def jacobi():
    for i in ti.grouped(x):
        b[i] = x[i] + F[i] * beta_dt**2
        F[i] = b[i] * 0
    Acc(F, b, 1)


@ti.func
def ballBoundReflect(pos, vel, center, radius):
    ret = vel
    above = tl.distance(pos, center) - radius
    if above <= 0:
        normal = tl.normalize(pos - center)
        NoV = tl.dot(vel, normal) - 6 * tl.smoothstep(above, 0, -0.1)
        if NoV < 0:
            ret -= NoV * normal
    return ret


@ti.kernel
def collide(x: ti.template(), v: ti.template()):
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
    #for i in ti.grouped(x):
    #    v[i] = tl.boundReflect(x[i], v[i], -1, 1)
    for i in ti.grouped(x):
        v[i] = ballBoundReflect(x[i], v[i], tl.vec(+0.0, +0.2, -0.0), 0.4)


@ti.kernel
def update_pos():
    for i in ti.grouped(x):
        x[i] = b[i]
        v[i] += F[i] * beta_dt


def implicit():
    prepare()
    for i in range(15):
        jacobi()
    collide(b, v)
    update_pos()


scene = t3.Scene()
model = t3.Model()
scene.add_model(model)

faces = t3.Face.var(N**2 * 2)
#lines = t3.Line.var(N**2 * 2)
vertices = t3.Vertex.var(N**2)
model.set_vertices(vertices)
model.add_geometry(faces)
#model.add_geometry(lines)


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
        faces[a * 2 + 0].idx = tl.vec(a, c, b)
        faces[a * 2 + 1].idx = tl.vec(a, d, c)
        #faces[a * 4 + 2].idx = tl.vec(b, a, d)
        #faces[a * 4 + 3].idx = tl.vec(b, d, c)
        #lines[a * 2 + 0].idx = tl.vec(a, b)
        #lines[a * 2 + 1].idx = tl.vec(a, d)


@ti.kernel
def update_display():
    for i in ti.grouped(x):
        vertices[i.dot(tl.vec(N, 1))].pos = x[i]


init()
init_display()
scene.set_light_dir([0.4, -1.5, -1.8])
print('[Hint] mouse drag to orbit camera')
with ti.GUI('Mass Spring') as gui:
    gui.frame = 0
    while gui.running and not gui.get_event(gui.ESCAPE):
        if not gui.is_pressed(gui.SPACE):
            for i in range(5):
                implicit()
            update_display()
        if gui.is_pressed(gui.LMB):
            scene.camera.from_mouse(gui)
        else:
            scene.camera.from_mouse([0.5, 0.4])

        scene.render()
        gui.set_image(scene.img)
        #gui.show(f'/tmp/{gui.frame:06d}.png')
        gui.show()
        gui.frame += 1
