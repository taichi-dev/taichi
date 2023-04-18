import numpy as np

import taichi as ti

ti.init(arch=ti.cpu)

num_vertices = 1000
pos = ti.Vector.field(3, dtype=ti.f32, shape=(10, 10, 10))
rgba = ti.Vector.field(4, dtype=ti.f32, shape=(10, 10, 10))


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
        rgba[i, j, k] = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random()])


place_pos()
series_prefix = "example.ply"
for frame in range(10):
    move_particles()
    fill_rgba()
    # now adding each channel only supports passing individual np.array
    # so converting into np.ndarray, reshape
    # remember to use a temp var to store so you dont have to convert back
    np_pos = np.reshape(pos.to_numpy(), (num_vertices, 3))
    np_rgba = np.reshape(rgba.to_numpy(), (num_vertices, 4))
    # create a PLYWriter
    writer = ti.tools.PLYWriter(num_vertices=num_vertices)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.add_vertex_rgba(np_rgba[:, 0], np_rgba[:, 1], np_rgba[:, 2], np_rgba[:, 3])
    writer.export_frame(frame, series_prefix)
