import argparse
import math
import os

import numpy as np

import taichi as ti

screen_res = (1000, 1000)

boundary_box_np = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
spawn_box_np = np.array([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]], dtype=np.float32)

particle_radius = 0.01
particle_diameter = particle_radius * 2
h = 4.0 * particle_radius
N_np = ((spawn_box_np[1] - spawn_box_np[0]) / particle_diameter + 1).astype(
    np.int32)
particle_num = N_np[0] * N_np[1] * N_np[2]

rest_density = 1000.0
mass = rest_density * particle_diameter * particle_diameter * particle_diameter * 0.8
pressure_scale = 10000.0
viscosity_scale = 0.1 * 3
tension_scale = 0.005
gamma = 1.0
substeps = 5
dt = 0.016 / substeps
eps = 1e-6
damping = 0.5
pi = math.pi


@ti.func
def W_poly6(R, h):
    r = R.norm()
    res = 0.0
    if r <= h:
        h2 = h * h
        h4 = h2 * h2
        h9 = h4 * h4 * h
        h2_r2 = h2 - r * r
        res = 315.0 / (64 * pi * h9) * h2_r2 * h2_r2 * h2_r2
    else:
        res = 0.0
    return res


@ti.func
def W_spiky_gradient(R, h):
    r = R.norm()
    res = ti.Vector([0.0, 0.0, 0.0])
    if r == 0.0:
        res = ti.Vector([0.0, 0.0, 0.0])
    elif r <= h:
        h3 = h * h * h
        h6 = h3 * h3
        h_r = h - r
        res = -45.0 / (pi * h6) * h_r * h_r * (R / r)
    else:
        res = ti.Vector([0.0, 0.0, 0.0])
    return res


W = W_poly6
W_gradient = W_spiky_gradient


@ti.kernel
def initialize(boundary_box: ti.any_arr(field_dim=1),
               spawn_box: ti.any_arr(field_dim=1), N: ti.any_arr(field_dim=1)):
    boundary_box[0] = [0.0, 0.0, 0.0]
    boundary_box[1] = [1.0, 1.0, 1.0]

    spawn_box[0] = [0.3, 0.3, 0.3]
    spawn_box[1] = [0.7, 0.7, 0.7]

    N[0] = 20
    N[1] = 20
    N[2] = 20


@ti.kernel
def initialize_particle(pos: ti.any_arr(field_dim=1),
                        spawn_box: ti.any_arr(field_dim=1),
                        N: ti.any_arr(field_dim=1),
                        gravity: ti.any_arr(field_dim=0)):
    gravity[None] = ti.Vector([0.0, -9.8, 0.0])
    for i in range(particle_num):
        pos[i] = (
            ti.Vector([i % N[0], i // N[0] % N[1], i // N[0] // N[1] % N[2]]) *
            particle_diameter + spawn_box[0])
        # print(i, pos[i], spawn_box[0], N[0], N[1], N[2])


@ti.kernel
def update_density(pos: ti.any_arr(field_dim=1), den: ti.any_arr(field_dim=1),
                   pre: ti.any_arr(field_dim=1)):
    for i in range(particle_num):
        den[i] = 0.0
        for j in range(particle_num):
            R = pos[i] - pos[j]
            den[i] += mass * W(R, h)
        pre[i] = pressure_scale * max(pow(den[i] / rest_density, gamma) - 1, 0)


@ti.kernel
def update_force(pos: ti.any_arr(field_dim=1), vel: ti.any_arr(field_dim=1),
                 den: ti.any_arr(field_dim=1), pre: ti.any_arr(field_dim=1),
                 acc: ti.any_arr(field_dim=1),
                 gravity: ti.any_arr(field_dim=0)):
    for i in range(particle_num):
        acc[i] = gravity[None]
        for j in range(particle_num):
            R = pos[i] - pos[j]

            acc[i] += (-mass * (pre[i] / (den[i] * den[i]) + pre[j] /
                                (den[j] * den[j])) * W_gradient(R, h))

            acc[i] += (viscosity_scale * mass * (vel[i] - vel[j]).dot(R) /
                       (R.norm() + 0.01 * h * h) / den[j] * W_gradient(R, h))

            R2 = R.dot(R)
            D2 = particle_diameter * particle_diameter
            if R2 > D2:
                acc[i] += -tension_scale * R * W(R, h)
            else:
                acc[i] += (
                    -tension_scale * R *
                    W(ti.Vector([0.0, 1.0, 0.0]) * particle_diameter, h))


@ti.kernel
def advance(pos: ti.any_arr(field_dim=1), vel: ti.any_arr(field_dim=1),
            acc: ti.any_arr(field_dim=1)):
    for i in range(particle_num):
        vel[i] += acc[i] * dt
        pos[i] += vel[i] * dt


@ti.kernel
def boundary_handle(pos: ti.any_arr(field_dim=1), vel: ti.any_arr(field_dim=1),
                    boundary_box: ti.any_arr(field_dim=1)):
    for i in range(particle_num):
        collision_normal = ti.Vector([0.0, 0.0, 0.0])
        for j in ti.static(range(3)):
            if pos[i][j] < boundary_box[0][j]:
                pos[i][j] = boundary_box[0][j]
                collision_normal[j] += -1.0
        for j in ti.static(range(3)):
            if pos[i][j] > boundary_box[1][j]:
                pos[i][j] = boundary_box[1][j]
                collision_normal[j] += 1.0
        collision_normal_length = collision_normal.norm()
        if collision_normal_length > eps:
            collision_normal /= collision_normal_length
            vel[i] -= (1.0 + damping) * collision_normal.dot(
                vel[i]) * collision_normal


@ti.kernel
def copy_data_from_ndarray_to_field(src: ti.template(), dst: ti.any_arr()):
    for I in ti.grouped(src):
        src[I] = dst[I]


parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    if args.arch == "cuda":
        arch = ti.cuda
    elif args.arch == "x64":
        arch = ti.x64
    elif args.arch == "vulkan":
        arch = ti.vulkan
    elif args.arch == "opengl":
        arch = ti.opengl
    else:
        assert False

    ti.init(arch=arch)

    # Initialize arrays
    N = ti.ndarray(
        ti.i32, shape=3
    )  # Potential bug: modify ti.f32 to ti.i32 leads to [all components of N are zeros].
    N.from_numpy(N_np)
    boundary_box = ti.Vector.ndarray(3, ti.f32, shape=2)
    boundary_box.from_numpy(boundary_box_np)
    spawn_box = ti.Vector.ndarray(3, ti.f32, shape=2)
    spawn_box.from_numpy(spawn_box_np)

    pos = ti.Vector.ndarray(3, ti.f32, shape=particle_num)
    vel = ti.Vector.ndarray(3, ti.f32, shape=particle_num)
    acc = ti.Vector.ndarray(3, ti.f32, shape=particle_num)
    den = ti.ndarray(ti.f32, shape=particle_num)
    pre = ti.ndarray(ti.f32, shape=particle_num)
    gravity = ti.Vector.ndarray(3, ti.f32, shape=())

    print('running in graph mode')

    # Serialize!
    mod = ti.aot.Module(arch)

    mod.add_kernel(initialize,
                   template_args={
                       'boundary_box': boundary_box,
                       'spawn_box': spawn_box,
                       'N': N
                   })
    mod.add_kernel(initialize_particle,
                   template_args={
                       'pos': pos,
                       'spawn_box': spawn_box,
                       'N': N,
                       'gravity': gravity
                   })
    mod.add_kernel(update_density,
                   template_args={
                       'pos': pos,
                       'den': den,
                       'pre': pre
                   })
    mod.add_kernel(update_force,
                   template_args={
                       'pos': pos,
                       'vel': vel,
                       'den': den,
                       'pre': pre,
                       'acc': acc,
                       'gravity': gravity
                   })
    mod.add_kernel(advance, template_args={'pos': pos, 'vel': vel, 'acc': acc})
    mod.add_kernel(boundary_handle,
                   template_args={
                       'pos': pos,
                       'vel': vel,
                       'boundary_box': boundary_box
                   })

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    tmpdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    mod.save(tmpdir, '')
