import taichi as ti
import taichi_glsl as tl
ti.init(arch=[ti.cuda, ti.metal])

dim = 3
N = 1024 * 8
dt = 2e-4
steps = 7
sun = ti.Vector([0.5, 0.5] if dim == 2 else [0.5, 0.5, 0.0])
gravity = 0.5
pressure = 0.3
pps_scale = 0.4
colorinit = 0.3
colordeg = 1.6
initvel = 0.07
res = 640

inv_m = ti.field(ti.f32)
color = ti.field(ti.f32)
x = ti.Vector.field(dim, ti.f32)
v = ti.Vector.field(dim, ti.f32)
ti.root.bitmasked(ti.i, N).place(x, v, inv_m, color)
xcnt = ti.field(ti.i32, ())
img = ti.field(ti.f32, (res, res))


@ti.kernel
def substep():
    ti.no_activate(x)
    for i in x:
        r = x[i] - sun
        ir = r / r.norm(1e-3)**3
        acci = pressure * ir * inv_m[i]
        acci += -gravity * ir
        v[i] += acci * dt

        x[i] += v[i] * dt
        color[i] *= ti.exp(-dt * colordeg)

        if not all(-0.1 <= x[i] <= 1.1):
            ti.deactivate(x.snode.parent(), [i])


@ti.kernel
def generate():
    r = x[0] - sun
    ir = 1 / r.norm(1e-3)**2
    pps = int(pps_scale * ir)
    for _ in range(pps):
        r = x[0]
        if ti.static(dim == 3):
            r = tl.randUnit3D()
        else:
            r = tl.randUnit2D()
        xi = ti.atomic_add(xcnt[None], 1) % (N - 1) + 1
        x[xi] = x[0]
        v[xi] = r * initvel + v[0]
        inv_m[xi] = 0.5 + ti.random()
        color[xi] = colorinit


@ti.kernel
def render():
    for p in ti.grouped(img):
        img[p] = 1e-6 / (p / res - ti.Vector([sun.x, sun.y])).norm(1e-4)**3
    for i in x:
        p = int(ti.Vector([x[i].x, x[i].y]) * res)
        img[p] += color[i]


inv_m[0] = 0
x[0].x = +0.5
x[0].y = -0.01
v[0].x = +0.6
v[0].y = +0.4
color[0] = 1

gui = ti.GUI('Comet', res)
while gui.running:
    gui.running = not gui.get_event(gui.ESCAPE)
    generate()
    for s in range(steps):
        substep()
    render()
    gui.set_image(img)
    gui.show()
