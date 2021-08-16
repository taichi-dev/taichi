import numpy as np

import taichi as ti

ti.init(ti.cuda)

N = 10

vertices = ti.Vector.field(2, ti.f32, N * 2)
colors = ti.Vector.field(3, ti.f32, N * 2)


@ti.kernel
def render_lines():
    for i in range(N):
        vertices[i * 2] = ti.Vector([i, i]) / N
        vertices[i * 2 + 1] = ti.Vector([i + 0.5, i + 0.5]) / N
        colors[i * 2] = ti.Vector([i, i, i]) / N
        colors[i * 2 + 1] = ti.Vector([i + 0.5, i + 0.5, i + 0.5]) / N


res = (1920, 1080)
window = ti.ui.Window("heyy", res)

frame_id = 0
canvas = window.get_canvas()

show_lines = True

use_per_vertex_colors = True

lines_color = (0, 0, 1)

width = 0.01

while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    render_lines()

    if show_lines:
        if use_per_vertex_colors:
            canvas.lines(vertices, per_vertex_color=colors, width=width)
        else:
            canvas.lines(vertices, color=lines_color, width=width)

    window.GUI.begin("hello window", 0.1, 0.1, 0.2, 0.8)
    window.GUI.text("hello text")
    show_lines = window.GUI.checkbox("show lines", show_lines)
    width = window.GUI.slider_float("width ", width, 0, 0.02)
    use_per_vertex_colors = window.GUI.checkbox("use per vertex colors",
                                                use_per_vertex_colors)
    if not use_per_vertex_colors:
        lines_color = window.GUI.color_edit_3("lines color", lines_color)

    window.GUI.end()

    window.show()
