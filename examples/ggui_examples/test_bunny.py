import pathlib

import numpy as np
import pywavefront

import taichi as ti

ti.init(ti.cpu)

scene = pywavefront.Wavefront(str(pathlib.Path(__file__).parent) +
                              "/bunny.obj",
                              collect_faces=True)

vertices_imported = scene.vertices
normals_imported = scene.parser.normals
faces_imported = scene.mesh_list[0].faces

num_vertices = len(vertices_imported)
#assert (num_vertices == len(normals_imported))
num_triangles = len(faces_imported)
num_indices = num_triangles * 3

vertices_host = []
normals_host = []
indices_host = []

for i in range(num_vertices):
    vertices_host += [vertices_imported[i]]
    normals_host += [normals_imported[i]]

for i in range(num_triangles):
    indices_host += faces_imported[i]

vertices_host = np.array(vertices_host)
normals_host = np.array(normals_host)
indices_host = np.array(indices_host)

vertices = ti.Vector.field(3, ti.f32, num_vertices)
normals = ti.Vector.field(3, ti.f32, num_vertices)
colors = ti.Vector.field(3, ti.f32, num_vertices)
indices = ti.field(ti.i32, num_indices)

print(np.min(faces_imported), np.max(faces_imported))


@ti.kernel
def copy_vertices(device: ti.template(), host: ti.ext_arr()):
    for i in device:
        device[i] = ti.Vector([host[i, 0], host[i, 1], host[i, 2]])


copy_vertices(vertices, vertices_host)


@ti.kernel
def copy_normals(device: ti.template(), host: ti.ext_arr()):
    for i in device:
        device[i] = ti.Vector([host[i, 0], host[i, 1], host[i, 2]])


copy_normals(normals, normals_host)


@ti.kernel
def copy_indices(device: ti.template(), host: ti.ext_arr()):
    for i in device:
        device[i] = host[i]


copy_indices(indices, indices_host)


@ti.kernel
def init_colors():
    for i in range(num_vertices):
        colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])


init_colors()

print(num_triangles, num_vertices, indices.shape, vertices.shape)

res = (1920, 1080)
window = ti.ui.Window("heyy", res)

frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

use_random_colors = False
with_normals = True

while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    camera.position(2, 2, 2)
    camera.lookat(0, 0, 0)
    camera.up(0, 1, 0)
    scene.set_camera(camera)

    if with_normals:
        if use_random_colors:
            scene.mesh(vertices,
                       indices,
                       per_vertex_color=colors,
                       normals=normals)
        else:
            scene.mesh(vertices, indices, color=(0, 0, 1), normals=normals)
    else:
        if use_random_colors:
            scene.mesh(
                vertices,
                indices,
                per_vertex_color=colors,
            )
        else:
            scene.mesh(vertices, indices, color=(0, 0, 1))

    scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

    canvas.scene(scene)

    window.GUI.begin("bunny settings", 0.1, 0.1, 0.2, 0.8)
    use_random_colors = window.GUI.checkbox("use_random_colors",
                                            use_random_colors)
    with_normals = window.GUI.checkbox("with_normals", with_normals)

    window.GUI.end()

    #
    window.show()
