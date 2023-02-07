import taichi as ti

ti.init(arch=ti.vulkan)


def test_geometry_2d():
    window = ti.ui.Window('test', (640, 480))
    canvas = window.get_canvas()

    # simple circles
    n_circles_0 = 10
    circle_positions_0 = ti.Vector.field(2, ti.f32, shape=n_circles_0)
    for i in range(n_circles_0):
        circle_positions_0[i] = ti.Vector([0.1, i * 0.1])

    # circles with per vertex colors
    n_circles_1 = 10
    circle_positions_1 = ti.Vector.field(2, ti.f32, shape=n_circles_1)
    circle_colors_1 = ti.Vector.field(3, ti.f32, shape=n_circles_1)
    for i in range(n_circles_0):
        circle_positions_1[i] = ti.Vector([0.2, i * 0.1])
        circle_colors_1[i] = ti.Vector([i * 0.1, 1.0 - i * 0.1, 0.5])

    # simple triangles
    n_triangles_0 = 10
    triangles_positions_0 = ti.Vector.field(2, ti.f32, shape=3 * n_triangles_0)
    for i in range(n_triangles_0):
        triangles_positions_0[3 * i] = ti.Vector([0.3, i * 0.1])
        triangles_positions_0[3 * i + 1] = ti.Vector([0.35, i * 0.1])
        triangles_positions_0[3 * i + 2] = ti.Vector([0.35, i * 0.1 + 0.05])

    # triangles with per vertex colors and indices
    triangles_positions_1 = ti.Vector.field(2, ti.f32, shape=4)
    triangles_colors_1 = ti.Vector.field(3, ti.f32, shape=4)
    triangles_positions_1[0] = ti.Vector([0.4, 0])
    triangles_positions_1[1] = ti.Vector([0.4, 1])
    triangles_positions_1[2] = ti.Vector([0.45, 0])
    triangles_positions_1[3] = ti.Vector([0.45, 1])
    triangles_colors_1[0] = ti.Vector([0, 0, 0])
    triangles_colors_1[1] = ti.Vector([1, 0, 0])
    triangles_colors_1[2] = ti.Vector([0, 1, 0])
    triangles_colors_1[3] = ti.Vector([1, 1, 0])
    triangle_indices_1 = ti.Vector.field(3, ti.i32, shape=2)
    triangle_indices_1[0] = ti.Vector([0, 1, 3])
    triangle_indices_1[1] = ti.Vector([0, 2, 3])

    # simple lines
    n_lines_0 = 10
    lines_positions_0 = ti.Vector.field(2, ti.f32, shape=2 * n_lines_0)
    for i in range(n_lines_0):
        lines_positions_0[2 * i] = ti.Vector([0.5, i * 0.1])
        lines_positions_0[2 * i + 1] = ti.Vector([0.5, i * 0.1 + 0.05])

    # lines with per vertex colors and indices
    lines_positions_1 = ti.Vector.field(2, ti.f32, shape=4)
    lines_colors_1 = ti.Vector.field(3, ti.f32, shape=4)
    lines_positions_1[0] = ti.Vector([0.6, 0])
    lines_positions_1[1] = ti.Vector([0.6, 1])
    lines_positions_1[2] = ti.Vector([0.65, 0])
    lines_positions_1[3] = ti.Vector([0.65, 1])
    lines_colors_1[0] = ti.Vector([0, 0, 0])
    lines_colors_1[1] = ti.Vector([1, 0, 0])
    lines_colors_1[2] = ti.Vector([0, 1, 0])
    lines_colors_1[3] = ti.Vector([1, 1, 0])
    lines_indices_1 = ti.Vector.field(2, ti.i32, shape=6)
    line_id = 0
    for i in range(4):
        for j in range(i + 1, 4):
            lines_indices_1[line_id] = ti.Vector([i, j])
            line_id += 1

    def render():

        canvas.circles(circle_positions_0, radius=0.05, color=(1, 0, 0))

        canvas.circles(circle_positions_1,
                       radius=0.05,
                       per_vertex_color=circle_colors_1)

        canvas.triangles(triangles_positions_0, color=(0, 0, 1))

        canvas.triangles(triangles_positions_1,
                         per_vertex_color=triangles_colors_1,
                         indices=triangle_indices_1)

        canvas.lines(lines_positions_0, width=0.01, color=(0, 1, 0))

        canvas.lines(lines_positions_1,
                     width=0.01,
                     per_vertex_color=lines_colors_1,
                     indices=lines_indices_1)

    # Render in off-line mode to check if there are errors
    for _ in range(300):
        render()
        window.show()

    render()
    ti.tools.imwrite(window.get_image_buffer_as_numpy(),
                     'test_geometry_2d.png')
    window.destroy()


test_geometry_2d()
