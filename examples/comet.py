import taichi as ti
import taichi_glsl as tl
ti.init()

dim = 3
N = 8192
NGps = 4
dt = 2e-3
steps = 3

inv_m = ti.field(ti.f32)
x = ti.Vector.field(dim, ti.f32)
v = ti.Vector.field(dim, ti.f32)
ti.root.bitmasked(ti.i, N).place(x, v, inv_m)
xcnt = ti.field(ti.i32)
ti.root.place(xcnt)


@ti.kernel
def substep():
    for i in x:
        acci = ti.Vector.unit(x.n, 1) * inv_m[i]
        v[i] += acci * dt

    ti.no_activate(x)
    for i in x:
        x[i] += v[i] * dt
        if not all(-0.1 <= x[i] <= 1.1):
            ti.deactivate(x.snode().parent(), [i])
            x[i].fill(0)
            v[i].fill(0)


@ti.kernel
def generate():
    for _ in ti.static(range(NGps)):
        r = x[0]
        if ti.static(x.n == 3):
            r = tl.randUnit3D()
        else:
            r = tl.randUnit2D()
        x[xcnt[None]] = r * 0.005 + x[0]
        v[xcnt[None]] = r * 0.1 + v[0]
        inv_m[xcnt[None]] = 0.75 + 0.5 * ti.random()
        xcnt[None] = xcnt[None] % (N - 1) + 1



inv_m[0] = 0
xcnt[None] = 1
x[0] = [0.5, 0.5, 0.0]

gui = ti.GUI('Comet')
while gui.running:
    gui.running = not gui.get_event(gui.ESCAPE)
    for s in range(steps):
        generate()
        substep()
    x_ = x.to_numpy()[:, :2]
    gui.circle(x_[0], radius=0.005 * 512, color=0xcceeff)
    gui.circles(x_[1:], radius=1, color=0xcceeff)
    gui.show()
