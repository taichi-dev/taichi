import os
import pathlib
import platform
import tempfile

import numpy as np
import pytest
from taichi._lib import core as _ti_core

import taichi as ti
from tests import test_utils

REGENERATE_GROUNDTRUTH_IMAGES = False
RENDER_REPEAT = 5
supported_archs = [ti.vulkan, ti.cuda]


def get_temp_png():
    f, name = tempfile.mkstemp(suffix='.png')
    os.close(f)
    return name


def write_temp_image(window):
    f = get_temp_png()
    window.write_image(f)
    try:
        os.remove(f)
    except OSError:
        pass


def verify_image(window, image_name, tolerence=0.1):
    if REGENERATE_GROUNDTRUTH_IMAGES:
        ground_truth_name = f"tests/python/expected/{image_name}.png"
        window.write_image(ground_truth_name)
    else:
        ground_truth_name = str(
            pathlib.Path(__file__).parent) + f"/expected/{image_name}.png"
        actual_name = get_temp_png()
        window.write_image(actual_name)
        ground_truth_np = ti.tools.imread(ground_truth_name)
        actual_np = ti.tools.imread(actual_name)
        assert len(ground_truth_np.shape) == len(actual_np.shape)
        for i in range(len(ground_truth_np.shape)):
            assert ground_truth_np.shape[i] == actual_np.shape[i]
        diff = ground_truth_np - actual_np
        mse = np.mean(diff * diff)
        assert mse <= tolerence  # the pixel values are 0~255
        os.remove(actual_name)


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_geometry_2d():
    window = ti.ui.Window('test', (640, 480), show_window=False)
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

    for _ in range(RENDER_REPEAT):
        render()
        write_temp_image(window)
    render()
    if (platform.system() == 'Darwin'):
        # FIXME: Use lower tolerence when macOS ggui supports wide lines
        verify_image(window, 'test_geometry_2d', 1.0)
    else:
        verify_image(window, 'test_geometry_2d')
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_geometry_3d():
    window = ti.ui.Window('test', (640, 480), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.make_camera()
    camera.position(0.0, 0.0, 1.5)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    # simple particles
    num_per_dim = 32
    num_particles_0 = int(num_per_dim**3)
    particles_positions_0 = ti.Vector.field(3, ti.f32, shape=num_particles_0)

    @ti.kernel
    def init_particles_0():
        for x, y, z in ti.ndrange(num_per_dim, num_per_dim, num_per_dim):
            i = x * (num_per_dim**2) + y * num_per_dim + z
            gap = 0.01
            particles_positions_0[i] = ti.Vector(
                [-0.4, 0, 0.0],
                dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap

    init_particles_0()

    # particles with individual colors
    num_per_dim = 32
    num_particles_1 = int(num_per_dim**3)
    particles_positions_1 = ti.Vector.field(3, ti.f32, shape=num_particles_1)
    particles_colors_1 = ti.Vector.field(3, ti.f32, shape=num_particles_1)

    @ti.kernel
    def init_particles_1():
        for x, y, z in ti.ndrange(num_per_dim, num_per_dim, num_per_dim):
            i = x * (num_per_dim**2) + y * num_per_dim + z
            gap = 0.01
            particles_positions_1[i] = ti.Vector(
                [0.2, 0, 0.0],
                dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap
            particles_colors_1[i] = ti.Vector([x, y, z],
                                              dt=ti.f32) / num_per_dim

    init_particles_1()

    # mesh
    vertices = ti.Vector.field(3, ti.f32, shape=8)
    colors = ti.Vector.field(3, ti.f32, shape=8)

    @ti.kernel
    def init_mesh():
        for i, j, k in ti.ndrange(2, 2, 2):
            index = i * 4 + j * 2 + k
            vertices[index] = ti.Vector(
                [-0.1, -0.3, 0.0],
                dt=ti.f32) + ti.Vector([i, j, k], dt=ti.f32) * 0.25
            colors[index] = ti.Vector([i, j, k], dt=ti.f32)

    init_mesh()
    indices = ti.field(ti.i32, shape=36)
    indices_np = np.array([
        0, 1, 2, 3, 1, 2, 4, 5, 6, 7, 5, 6, 0, 1, 4, 5, 1, 4, 2, 3, 6, 7, 3, 6,
        0, 2, 4, 6, 2, 4, 1, 3, 5, 7, 3, 5
    ],
                          dtype=np.int32)
    indices.from_numpy(indices_np)

    def render():
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

        scene.particles(particles_positions_0, radius=0.01, color=(0.5, 0, 0))

        scene.particles(particles_positions_1,
                        radius=0.01,
                        per_vertex_color=particles_colors_1)

        scene.mesh(vertices,
                   per_vertex_color=colors,
                   indices=indices,
                   two_sided=True)

        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        write_temp_image(window)
    render()
    verify_image(window, 'test_geometry_3d')
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_set_image():
    window = ti.ui.Window('test', (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.Vector.field(4, ti.f32, (512, 512))

    @ti.kernel
    def init_img():
        for i, j in img:
            img[i, j] = ti.Vector([i, j, 0, 512], dt=ti.f32) / 512

    init_img()

    def render():
        canvas.set_image(img)

    for _ in range(RENDER_REPEAT):
        render()
        write_temp_image(window)
    render()
    verify_image(window, 'test_set_image')
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_imgui():
    window = ti.ui.Window('test', (640, 480), show_window=False)

    def render():
        with window.GUI.sub_window("window 0", 0.1, 0.1, 0.8, 0.2) as w:
            w.text("Hello Taichi!")
            w.text("Hello Again!")
        with window.GUI.sub_window("window 1", 0.1, 0.4, 0.8, 0.2) as w:
            w.button("Press to unlease creativity")
            w.slider_float('creativity level', 100.0, 0.0, 100.0)
        with window.GUI.sub_window("window 2", 0.1, 0.7, 0.8, 0.2) as w:
            w.color_edit_3('Heyy', (0, 0, 1))

    for _ in range(RENDER_REPEAT):
        render()
        write_temp_image(window)
    render()
    verify_image(window, 'test_imgui')
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_exit_without_showing():
    window = ti.ui.Window("Taichi", (256, 256), show_window=False)
