import taichi as ti
import random
import os
ti.init(arch=ti.cpu)

dim = 3
n_particles = 8192

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)

for i in range(n_particles):
    x[i] = [random.random() * 0.4 + 0.2, random.random() *
            0.4 + 0.2, random.random() * 0.4 + 0.2]
x = x.to_numpy()
writer = ti.PLYWriter(num_vertices=n_particles)
writer.add_vertex_pos(x[:, 0], x[:, 1], x[:, 2])

# export binary ply file
writer.export("example.ply")
os.remove("example.ply")
# export ascii ply file
writer.export_ascii("example_ascii.ply")
os.remove("example_ascii.ply")
