import argparse
import math
import os

from taichi.lang.impl import grouped

import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str)
args = parser.parse_args()

if args.arch == "cuda":
    arch = ti.cuda
elif args.arch == "x64":
    arch = ti.x64
else:
    assert False

ti.init(arch=arch)

dim = 3
N = 1024 * 8
dt = 2e-4
steps = 7
sun = ti.Vector([0.5, 0.5, 0.0])
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

sym_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                       'arr',
                       ti.f32,
                       field_dim=3,
                       element_shape=())
img_c = 4


@ti.kernel
def img_to_ndarray(arr: ti.types.ndarray()):
    for I in grouped(img):
        for c in range(img_c):
            arr[I, c] = img[I]


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
        r_sq_inverse = r / r.norm(1e-3)**3
        acceleration = (pressure * inv_m[i] - gravity) * r_sq_inverse
        v[i] += acceleration * dt
        x[i] += v[i] * dt
        color[i] *= ti.exp(-dt * color_decay)

        if not all(-0.1 <= x[i] <= 1.1):
            ti.deactivate(x.snode.parent(), [i])


@ti.kernel
def generate():
    r = x[0] - sun
    n_tail_paticles = int(tail_paticle_scale / r.norm(1e-3)**2)
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
        img[p] = 1e-6 / (p / res - ti.Vector([sun.x, sun.y])).norm(1e-4)**3
    for i in x:
        p = int(ti.Vector([x[i].x, x[i].y]) * res)
        img[p] += color[i]


@ti.kernel
def initialize():
    inv_m[0] = 0
    x[0].x = +0.5
    x[0].y = -0.01
    v[0].x = +0.6
    v[0].y = +0.4
    color[0] = 1


def save_kernels(arch):
    mod = ti.aot.Module(arch)

    # Initialize
    g_init_builder = ti.graph.GraphBuilder()
    g_init_builder.dispatch(initialize)

    # Update Per Iter
    g_update_builder = ti.graph.GraphBuilder()
    g_update_builder.dispatch(generate)

    substep_builder = g_update_builder.create_sequential()
    substep_builder.dispatch(substep)
    for i in range(steps):
        g_update_builder.append(substep_builder)
    g_update_builder.dispatch(render)
    g_update_builder.dispatch(img_to_ndarray, sym_arr)

    # Compile to Graph
    g_init = g_init_builder.compile()
    g_update = g_update_builder.compile()

    mod.add_graph('init', g_init)
    mod.add_graph('update', g_update)

    mod.add_field("inv_m", inv_m)
    mod.add_field("color", color)
    mod.add_field("x", x)
    mod.add_field("v", v)
    mod.add_field("count", count)
    mod.add_field("img", img)

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    tmpdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    mod.save(tmpdir, 'whatever')


if __name__ == '__main__':
    save_kernels(arch=arch)
