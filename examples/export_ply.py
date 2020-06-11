import taichi as ti
import numpy as np
import random
import os

ti.init(arch=ti.cpu)

x = ti.Vector(3, dt=ti.f32, shape=(10, 10, 10))
color = ti.var(ti.f32, shape=(10, 10, 10, 3))


@ti.kernel
def place_pos():
    for i, j, k in x:
        x[i, j, k][0] = 0.1*i
        x[i, j, k][1] = 0.1*j
        x[i, j, k][2] = 0.1*k


@ti.kernel
def fill_color():
    for I in ti.grouped(color):
        color[I] = random.random() * 255


place_pos()
fill_color()
writer = ti.PLYWriter(num_vertices=1000)
writer.add_vertex_pos(x)
writer.add_vertex_color(color)

# export binary ply file
writer.export("example.ply")
# os.remove("example.ply")
# export ascii ply file
writer.export_ascii("example_ascii.ply")
# os.remove("example_ascii.ply")
