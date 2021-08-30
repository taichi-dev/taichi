import sys

import numpy as np

import taichi as ti

ti.init(arch=ti.cuda)

### Parameters

N = 128
W = 1
L = W / N
gravity = 0.5
stiffness = 1600
damping = 2
steps = 30
dt = 5e-4

num_balls = 1
ball_radius = 0.3
ball_centers = ti.Vector.field(3, float, num_balls)

x = ti.Vector.field(3, float, (N, N))
v = ti.Vector.field(3, float, (N, N))

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, N * N)


@ti.kernel
def init():
    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector([(i + 0.5) * L - 0.5, (j + 0.5) * L / ti.sqrt(2),
                             (N - j) * L / ti.sqrt(2)])

        if i < N - 1 and j < N - 1:
            tri_id = ((i * (N - 1)) + j) * 2
            indices[tri_id * 3] = i * N + j
            indices[tri_id * 3 + 1] = (i + 1) * N + j
            indices[tri_id * 3 + 2] = i * N + (j + 1)

            tri_id += 1
            indices[tri_id * 3] = (i + 1) * N + j + 1
            indices[tri_id * 3 + 1] = i * N + (j + 1)
            indices[tri_id * 3 + 2] = (i + 1) * N + j
    ball_centers[0] = ti.Vector([0.0, -0.5, -0.0])


links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [ti.Vector([*v]) for v in links]


@ti.func
def reflect(v, normal):
    return v.dot(normal) * 2 - v


@ti.func
def ballBoundReflect(pos, vel, center, radius, bounce_magnitude=0.1):
    ret = vel
    distance = (pos - center).norm()
    if distance <= radius:
        normal = (pos - radius).normalized()
        ret = -1 * bounce_magnitude * reflect(vel, normal)
    return ret


@ti.kernel
def substep():
    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[min(max(i + d, 0), ti.Vector([N - 1, N - 1]))] - x[i]
            length = L * float(d).norm()
            acc += disp * (disp.norm() - length) / length**2
        v[i] += stiffness * acc * dt
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
        for b in range(num_balls):
            v[i] = ballBoundReflect(x[i], v[i], ball_centers[b],
                                    ball_radius * 1.2)
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        x[i] += dt * v[i]


@ti.kernel
def update_verts():
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]


init()

window = ti.ui.Window("Cloth", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

while window.running:
    update_verts()

    for i in range(steps):
        substep()

    camera.position(0, 0, 3)
    camera.lookat(0, 0, 0)
    camera.up(0, 1, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.mesh(vertices, indices=indices, color=(0.5, 0.5, 0.5))
    scene.particles(ball_centers, radius=ball_radius, color=(0.5, 0, 0))
    canvas.scene(scene)
    window.show()
