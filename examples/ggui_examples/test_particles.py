import numpy as np

import taichi as ti

ti.init(ti.cuda)

num_vertices = 10

vertices = ti.Vector.field(3, ti.f32, num_vertices)


@ti.kernel
def render_particles():
    for i in vertices:
        vertices[i] = ti.Vector([i, i, i]) / num_vertices


res = (1920, 1080)
window = ti.ui.Window("heyy", res)

frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

show_particles = True
camera_x = 2.0
camera_y = 2.0
camera_z = 2.0

particles_color = (0, 0, 1)

while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    render_particles()

    camera.position(camera_x, camera_y, camera_z)
    camera.lookat(0, 0, 0)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    if show_particles:
        scene.particles(vertices, color=particles_color, radius=0.1)
    scene.point_light(pos=(0, 10, 0), color=(1, 1, 1))

    canvas.scene(scene)

    window.GUI.begin("hello window", 0.1, 0.1, 0.2, 0.8)
    window.GUI.text("hello text")
    show_particles = window.GUI.checkbox("show particles", show_particles)
    camera_x = window.GUI.slider_float("camera x", camera_x, -10, 10)
    camera_y = window.GUI.slider_float("camera y", camera_y, -10, 10)
    camera_z = window.GUI.slider_float("camera z", camera_z, -10, 10)
    particles_color = window.GUI.color_edit_3("particles color",
                                              particles_color)
    if window.GUI.button("heyy"):
        print("hey")
    window.GUI.end()

    #
    window.show()
