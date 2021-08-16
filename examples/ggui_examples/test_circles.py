import numpy as np

import taichi as ti

ti.init(ti.cuda)

num_vertices = 10

vertices = ti.Vector.field(2, ti.f32, num_vertices)


@ti.kernel
def render_circles():
    for i in vertices:
        vertices[i] = ti.Vector([i, i]) / num_vertices


res = (1920, 1080)
window = ti.ui.Window("heyy", res)

frame_id = 0
canvas = window.get_canvas()

show_circles = True

circles_color = (0, 0, 1)

radius = 0.1

while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    render_circles()

    if show_circles:
        canvas.circles(vertices, color=circles_color, radius=radius)

    window.GUI.begin("hello window", 0.1, 0.1, 0.2, 0.8)
    window.GUI.text("hello text")
    show_circles = window.GUI.checkbox("show circles", show_circles)
    circles_color = window.GUI.color_edit_3("circles color", circles_color)
    radius = window.GUI.slider_float("particles radius ", radius, 0, 2)

    window.GUI.end()

    window.show()
