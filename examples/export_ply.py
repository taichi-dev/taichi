import taichi as ti
import numpy as np
import random
import os

ti.init(arch=ti.cpu)

pos = ti.Vector(3, dt=ti.f32, shape=(10, 10, 10))
rgba = ti.Vector(4, dt=ti.f32, shape=(10, 10, 10))


@ti.kernel
def place_pos():
    for i, j, k in pos:
        pos[i, j, k] = 0.1 * ti.Vector([i, j, k])


@ti.kernel
def move_particles():
    for i, j, k in pos:
        pos[i, j, k] += ti.Vector([0.1, 0.1, 0.1])


@ti.kernel
def fill_rgba():
    for i, j, k in rgba:
        rgba[i, j, k] = ti.Vector(
            [ti.random(), ti.random(), ti.random(), ti.random()])


num_vertices = 1000
place_pos()
fill_rgba()
series_prefix = "example.ply"
for frame in range(10):
    move_particles()
    fill_rgba()
    writer = ti.PLYWriter(num_vertices=num_vertices)
    writer.add_vertex_pos(pos)
    writer.add_vertex_color(rgba)
    writer.export_frame(frame, series_prefix)
