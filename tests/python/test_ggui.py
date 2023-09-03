import platform

import numpy as np
import pytest
from taichi._lib import core as _ti_core

import taichi as ti
from tests import test_utils
from tests.test_utils import verify_image

# FIXME: render(); get_image_buffer_as_numpy() loop does not actually redraw
RENDER_REPEAT = 5
# FIXME: enable ggui tests on ti.cpu backend. It's blocked by macos10.15
supported_archs = [ti.vulkan, ti.cuda, ti.metal]


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_geometry_2d():
    window = ti.ui.Window("test", (640, 480), show_window=False)
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

    # circles with per vertex radius
    n_circles_2 = 10
    circle_positions_2 = ti.Vector.field(2, ti.f32, shape=n_circles_2)
    circle_radii_2 = ti.field(ti.f32, shape=n_circles_2)
    for i in range(n_circles_2):
        circle_positions_2[i] = ti.Vector([0.75, i * 0.1])
        circle_radii_2[i] = (i + 1) / n_circles_2 * 0.05

    def render():
        canvas.circles(circle_positions_0, radius=0.05, color=(1, 0, 0))

        canvas.circles(circle_positions_1, radius=0.05, per_vertex_color=circle_colors_1)

        canvas.triangles(triangles_positions_0, color=(0, 0, 1))

        canvas.triangles(
            triangles_positions_1,
            per_vertex_color=triangles_colors_1,
            indices=triangle_indices_1,
        )

        canvas.lines(lines_positions_0, width=0.01, color=(0, 1, 0))

        canvas.lines(
            lines_positions_1,
            width=0.01,
            per_vertex_color=lines_colors_1,
            indices=lines_indices_1,
        )

        canvas.circles(circle_positions_2, radius=0.05, color=(0, 0, 1), per_vertex_radius=circle_radii_2)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_geometry_2d", tolerance=0.05)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_geometry_3d():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
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
            particles_positions_0[i] = ti.Vector([-0.4, 0, 0.0], dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap

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
            particles_positions_1[i] = ti.Vector([0.2, 0, 0.0], dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap
            particles_colors_1[i] = ti.Vector([x, y, z], dt=ti.f32) / num_per_dim

    init_particles_1()

    # mesh
    vertices = ti.Vector.field(3, ti.f32, shape=8)
    colors = ti.Vector.field(3, ti.f32, shape=8)

    @ti.kernel
    def init_mesh():
        for i, j, k in ti.ndrange(2, 2, 2):
            index = i * 4 + j * 2 + k
            vertices[index] = ti.Vector([-0.1, -0.3, 0.0], dt=ti.f32) + ti.Vector([i, j, k], dt=ti.f32) * 0.25
            colors[index] = ti.Vector([i, j, k], dt=ti.f32)

    init_mesh()
    indices = ti.field(ti.i32, shape=36)
    indices_np = np.array(
        [
            0,
            1,
            2,
            3,
            1,
            2,
            4,
            5,
            6,
            7,
            5,
            6,
            0,
            1,
            4,
            5,
            1,
            4,
            2,
            3,
            6,
            7,
            3,
            6,
            0,
            2,
            4,
            6,
            2,
            4,
            1,
            3,
            5,
            7,
            3,
            5,
        ],
        dtype=np.int32,
    )
    indices.from_numpy(indices_np)

    def render():
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

        scene.particles(particles_positions_0, radius=0.01, color=(0.5, 0, 0))

        scene.particles(particles_positions_1, radius=0.01, per_vertex_color=particles_colors_1)

        scene.mesh(vertices, per_vertex_color=colors, indices=indices, two_sided=True)

        canvas.scene(scene)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_geometry_3d")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_geometry_3d_old():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
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
            particles_positions_0[i] = ti.Vector([-0.4, 0, 0.0], dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap

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
            particles_positions_1[i] = ti.Vector([0.2, 0, 0.0], dt=ti.f32) + ti.Vector([x, y, z], dt=ti.f32) * gap
            particles_colors_1[i] = ti.Vector([x, y, z], dt=ti.f32) / num_per_dim

    init_particles_1()

    # mesh
    vertices = ti.Vector.field(3, ti.f32, shape=8)
    colors = ti.Vector.field(3, ti.f32, shape=8)

    @ti.kernel
    def init_mesh():
        for i, j, k in ti.ndrange(2, 2, 2):
            index = i * 4 + j * 2 + k
            vertices[index] = ti.Vector([-0.1, -0.3, 0.0], dt=ti.f32) + ti.Vector([i, j, k], dt=ti.f32) * 0.25
            colors[index] = ti.Vector([i, j, k], dt=ti.f32)

    init_mesh()
    indices = ti.field(ti.i32, shape=36)
    indices_np = np.array(
        [
            0,
            1,
            2,
            3,
            1,
            2,
            4,
            5,
            6,
            7,
            5,
            6,
            0,
            1,
            4,
            5,
            1,
            4,
            2,
            3,
            6,
            7,
            3,
            6,
            0,
            2,
            4,
            6,
            2,
            4,
            1,
            3,
            5,
            7,
            3,
            5,
        ],
        dtype=np.int32,
    )
    indices.from_numpy(indices_np)

    def render():
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))

        scene.particles(particles_positions_0, radius=0.01, color=(0.5, 0, 0))

        scene.particles(particles_positions_1, radius=0.01, per_vertex_color=particles_colors_1)

        scene.mesh(vertices, per_vertex_color=colors, indices=indices, two_sided=True)

        canvas.scene(scene)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_geometry_3d")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_3d_lines():
    N = 4
    CAMERA_POS = ti.Vector([-3, 2.5, -4])

    coor_pos = ti.Vector.field(3, dtype=ti.f32, shape=N * N * N)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=N * N * N)
    coor_idx = ti.ndarray(dtype=ti.i32, shape=N * N * N * 2)  # not works

    # Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
    @ti.func
    def compact_1by2(input: ti.u32):
        x = input & 0x09249249  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        x = (x ^ (x >> 2)) & 0x030C30C3  # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x >> 4)) & 0x0300F00F  # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x >> 8)) & 0x7F0000FF  # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x >> 16)) & 0x000003FF  # x = ---- ---- ---- ---- ---- --98 7654 3210
        return x

    @ti.func
    def decode_morton(code: ti.u32):
        return ti.Vector([compact_1by2(code >> 0), compact_1by2(code >> 1), compact_1by2(code >> 2)])

    @ti.kernel
    def init_coordinates(coor_idx: ti.types.ndarray()):
        for i, j, k in ti.ndrange(N, N, N):
            idx = i * N * N + j * N + k
            coor_pos[idx] = ti.Vector([i, j, k])
            fpos = ti.cast(ti.Vector([i, j, k]), ti.f32) + 0.01
            colors[idx] = fpos.normalized() * (0.1 + 2.0 / (ti.math.distance(fpos, CAMERA_POS) - 3.0))
        for i in ti.ndrange(N * N * N - 1):
            ipos0 = decode_morton(i)
            lindex0 = ipos0.x * N * N + ipos0.y * N + ipos0.z
            ipos1 = decode_morton(i + 1)
            lindex1 = ipos1.x * N * N + ipos1.y * N + ipos1.z
            coor_idx[i * 2 + 0] = lindex0
            coor_idx[i * 2 + 1] = lindex1

    window = ti.ui.Window("Test for Drawing 3d-lines", (512, 512), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()

    def render():
        init_coordinates(coor_idx)
        camera = ti.ui.Camera()
        camera.position(CAMERA_POS.x, CAMERA_POS.y, CAMERA_POS.z)
        camera.lookat(2, 1, 2)
        scene.set_camera(camera)
        scene.lines(coor_pos, indices=coor_idx, per_vertex_color=colors, width=3.0)
        canvas.scene(scene)

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_3d_lines")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_3d_lines_old():
    N = 4
    CAMERA_POS = ti.Vector([-3, 2.5, -4])

    coor_pos = ti.Vector.field(3, dtype=ti.f32, shape=N * N * N)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=N * N * N)
    coor_idx = ti.ndarray(dtype=ti.i32, shape=N * N * N * 2)  # not works

    # Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
    @ti.func
    def compact_1by2(input: ti.u32):
        x = input & 0x09249249  # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        x = (x ^ (x >> 2)) & 0x030C30C3  # x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x >> 4)) & 0x0300F00F  # x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x >> 8)) & 0x7F0000FF  # x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x >> 16)) & 0x000003FF  # x = ---- ---- ---- ---- ---- --98 7654 3210
        return x

    @ti.func
    def decode_morton(code: ti.u32):
        return ti.Vector([compact_1by2(code >> 0), compact_1by2(code >> 1), compact_1by2(code >> 2)])

    @ti.kernel
    def init_coordinates(coor_idx: ti.types.ndarray()):
        for i, j, k in ti.ndrange(N, N, N):
            idx = i * N * N + j * N + k
            coor_pos[idx] = ti.Vector([i, j, k])
            fpos = ti.cast(ti.Vector([i, j, k]), ti.f32) + 0.01
            colors[idx] = fpos.normalized() * (0.1 + 2.0 / (ti.math.distance(fpos, CAMERA_POS) - 3.0))
        for i in ti.ndrange(N * N * N - 1):
            ipos0 = decode_morton(i)
            lindex0 = ipos0.x * N * N + ipos0.y * N + ipos0.z
            ipos1 = decode_morton(i + 1)
            lindex1 = ipos1.x * N * N + ipos1.y * N + ipos1.z
            coor_idx[i * 2 + 0] = lindex0
            coor_idx[i * 2 + 1] = lindex1

    window = ti.ui.Window("Test for Drawing 3d-lines", (512, 512), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()

    def render():
        init_coordinates(coor_idx)
        camera = ti.ui.Camera()
        camera.position(CAMERA_POS.x, CAMERA_POS.y, CAMERA_POS.z)
        camera.lookat(2, 1, 2)
        scene.set_camera(camera)
        scene.lines(coor_pos, indices=coor_idx, per_vertex_color=colors, width=3.0)
        canvas.scene(scene)

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_3d_lines")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_set_image():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.Vector.field(4, ti.f32, (512, 512))

    @ti.kernel
    def init_img():
        for i, j in img:
            img[i, j] = ti.Vector([i, j, 0, 512], dt=ti.f32) / 512

    init_img()

    def render():
        canvas.set_image(img)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()

    verify_image(window.get_image_buffer_as_numpy(), "test_set_image")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_set_image_flat_field():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.field(ti.f32, (512, 512, 4))

    @ti.kernel
    def init_img():
        for i, j in ti.ndrange(img.shape[0], img.shape[1]):
            img[i, j, 0] = i / 512
            img[i, j, 1] = j / 512
            img[i, j, 2] = 0
            img[i, j, 3] = 1.0

    init_img()

    def render():
        canvas.set_image(img)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()

    verify_image(window.get_image_buffer_as_numpy(), "test_set_image")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.vulkan, ti.metal])
def test_set_image_with_texture():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.Texture(ti.Format.rgba8, (512, 512))

    @ti.kernel
    def init_img(img: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0)):
        for i, j in ti.ndrange(512, 512):
            img.store(ti.Vector([i, j]), ti.Vector([i, j, 0, 512], dt=ti.f32) / 512)

    init_img(img)

    def render():
        canvas.set_image(img)

    # Render in off-line mode to check if there are errors
    for _ in range(3):
        render()
        window.get_image_buffer_as_numpy()

    render()

    # Relaxed error because texture sampler differences
    # Note: the error is measured from a 0..255 range image
    verify_image(window.get_image_buffer_as_numpy(), "test_set_image", 0.3)
    window.destroy()


# NOTE: Cannot automate the test for the case of ImGui scaling on HiDPI displays. So that needs to be tested manually.
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_imgui():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    gui = window.get_gui()

    def render():
        with gui.sub_window("window 0", 0.1, 0.1, 0.8, 0.2) as w:
            w.text("Hello Taichi!")
            w.text("Hello Again!")
        with gui.sub_window("window 1", 0.1, 0.4, 0.8, 0.2) as w:
            w.button("Press to unlease creativity")
            w.slider_float("creativity level", 100.0, 0.0, 100.0)
        with gui.sub_window("window 2", 0.1, 0.7, 0.8, 0.2) as w:
            w.color_edit_3("Heyy", (0, 0, 1))

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_imgui")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_exit_without_showing():
    window = ti.ui.Window("Taichi", (256, 256), show_window=False)


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_get_camera_view_and_projection_matrix():
    window = ti.ui.Window("Taichi", (256, 256), show_window=False)
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 3)
    camera.lookat(0, 0, 0)

    scene.set_camera(camera)

    view_matrix = camera.get_view_matrix()
    projection_matrix = camera.get_projection_matrix(1080 / 720)

    for i in range(4):
        assert abs(view_matrix[i, i] - 1) <= 1e-5
    assert abs(view_matrix[3, 2] + 3) <= 1e-5

    assert abs(projection_matrix[0, 0] - 1.6094756) <= 1e-5
    assert abs(projection_matrix[1, 1] - 2.4142134) <= 1e-5
    assert abs(projection_matrix[2, 2] - 1.0001000e-4) <= 1e-5
    assert abs(projection_matrix[2, 3] + 1.0000000) <= 1e-5
    assert abs(projection_matrix[3, 2] - 1.0001000e-1) <= 1e-5
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_get_camera_view_and_projection_matrix_old():
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 3)
    camera.lookat(0, 0, 0)

    scene.set_camera(camera)

    view_matrix = camera.get_view_matrix()
    projection_matrix = camera.get_projection_matrix(1080 / 720)

    for i in range(4):
        assert abs(view_matrix[i, i] - 1) <= 1e-5
    assert abs(view_matrix[3, 2] + 3) <= 1e-5

    assert abs(projection_matrix[0, 0] - 1.6094756) <= 1e-5
    assert abs(projection_matrix[1, 1] - 2.4142134) <= 1e-5
    assert abs(projection_matrix[2, 2] - 1.0001000e-4) <= 1e-5
    assert abs(projection_matrix[2, 3] + 1.0000000) <= 1e-5
    assert abs(projection_matrix[3, 2] - 1.0001000e-1) <= 1e-5


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_fetching_color_attachment():
    window = ti.ui.Window("test", (640, 480), show_window=False)
    canvas = window.get_canvas()

    img = ti.Vector.field(4, ti.f32, (512, 512))

    @ti.kernel
    def init_img():
        for i, j in img:
            img[i, j] = ti.Vector([i, j, 0, 512], dt=ti.f32) / 512

    init_img()

    def render():
        canvas.set_image(img)

    # Render in off-line mode to check if there are errors
    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_set_image")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_fetching_depth_attachment():
    window = ti.ui.Window("test", (512, 512), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
    ball_center[0] = ti.math.vec3(0, 0, 0.5)

    def render():
        camera.position(0.0, 0.0, 1)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(ball_center, radius=0.05, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_depth_buffer_as_numpy(), "test_depth")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_fetching_depth_attachment_old():
    window = ti.ui.Window("test", (512, 512), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
    ball_center[0] = ti.math.vec3(0, 0, 0.5)

    def render():
        camera.position(0.0, 0.0, 1)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(ball_center, radius=0.05, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_depth_buffer_as_numpy(), "test_depth")
    window.destroy()


@pytest.mark.parametrize("offset", [None, (0, 0), (-256, -256), (256, -256), (-256, 256), (256, 256), (23333, 233333)])
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_get_depth_buffer_with_offset(offset):
    window = ti.ui.Window("test", (512, 512), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
    ball_center[0] = ti.math.vec3(0, 0, 0.5)

    def render():
        camera.position(0.0, 0.0, 1)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(ball_center, radius=0.05, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()

    depth_buffer_field = ti.field(dtype=ti.f32, shape=(512, 512), offset=offset)
    window.get_depth_buffer(depth_buffer_field)
    verify_image(depth_buffer_field, "test_depth")
    window.destroy()


@pytest.mark.parametrize("offset", [None, (0, 0), (-256, -256), (256, -256), (-256, 256), (256, 256), (23333, 233333)])
@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_get_depth_buffer_with_offset_old(offset):
    window = ti.ui.Window("test", (512, 512), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
    ball_center[0] = ti.math.vec3(0, 0, 0.5)

    def render():
        camera.position(0.0, 0.0, 1)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(ball_center, radius=0.05, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()

    depth_buffer_field = ti.field(dtype=ti.f32, shape=(512, 512), offset=offset)
    window.get_depth_buffer(depth_buffer_field)
    verify_image(depth_buffer_field, "test_depth")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_lines():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(points_pos, color=(0.28, 0.68, 0.99), width=5.0)
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_lines")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_lines_old():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(points_pos, color=(0.28, 0.68, 0.99), width=5.0)
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_lines")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_particles():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(
            particles_pos,
            color=(0.68, 0.26, 0.19),
            radius=0.5,
            index_offset=2,
            index_count=6,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_particles")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_particles_old():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(
            particles_pos,
            color=(0.68, 0.26, 0.19),
            radius=0.5,
            index_offset=2,
            index_count=6,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_particles")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_particles_per_vertex_rad_and_col():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    particles_col = ti.Vector.field(3, dtype=ti.f32, shape=N)
    particles_radii = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    @ti.kernel
    def init_points_col(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [(i + 1) / N, 0.5, (i + 1) / N]

    @ti.kernel
    def init_points_radii(radii: ti.template()):
        for i in range(radii.shape[0]):
            radii[i] = (i + 1) * 0.05

    init_points_pos(particles_pos)
    init_points_radii(particles_radii)
    init_points_col(particles_col)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(
            particles_pos,
            color=(0.68, 0.26, 0.19),
            radius=0.5,
            per_vertex_color=particles_col,
            per_vertex_radius=particles_radii,
            index_offset=2,
            index_count=6,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_particles_per_vertex_rad_and_col")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_particles_per_vertex_rad_and_col_old():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    particles_col = ti.Vector.field(3, dtype=ti.f32, shape=N)
    particles_radii = ti.field(dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    @ti.kernel
    def init_points_col(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [(i + 1) / N, 0.5, (i + 1) / N]

    @ti.kernel
    def init_points_radii(radii: ti.template()):
        for i in range(radii.shape[0]):
            radii[i] = (i + 1) * 0.05

    init_points_pos(particles_pos)
    init_points_radii(particles_radii)
    init_points_col(particles_col)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(
            particles_pos,
            color=(0.68, 0.26, 0.19),
            radius=0.5,
            per_vertex_color=particles_col,
            per_vertex_radius=particles_radii,
            index_offset=2,
            index_count=6,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_particles_per_vertex_rad_and_col")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_mesh():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.5, 1, -1)
    camera.lookat(1, 0.5, 0)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            index_count=2 * NT,
            index_offset=9,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_mesh")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_mesh_old():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(1.5, 1, -1)
    camera.lookat(1, 0.5, 0)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            index_count=2 * NT,
            index_offset=9,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_mesh")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_lines():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(
            points_pos,
            color=(0.28, 0.68, 0.99),
            width=5.0,
            vertex_count=6,
            vertex_offset=2,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_lines", 0.3)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_lines_old():
    N = 10
    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points_pos(points: ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test", (768, 768), show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 5, -10)
    camera.lookat(3, 3, 1)

    def render():
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(
            points_pos,
            color=(0.28, 0.68, 0.99),
            width=5.0,
            vertex_count=6,
            vertex_offset=2,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_lines", 0.3)
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_mesh_instances():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    # Instance Attribute Information
    NInstanceRows = 100
    NInstanceCols = 100
    NInstance = NInstanceRows * NInstanceCols
    instances_transforms = ti.Matrix.field(4, 4, ti.f32, shape=(NInstance,))

    @ti.kernel
    def init_transforms_of_instances():
        identity = ti.Matrix.identity(ti.f32, 4)
        for i in range(NInstanceRows):
            for j in range(NInstanceCols):
                index = i * NInstanceCols + j
                instances_transforms[index] = identity
                translate_matrix = ti.math.translate(1.2 * j, 0, -1.2 * i)
                instances_transforms[index] = translate_matrix @ instances_transforms[index]

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    @ti.kernel
    def update_transform(t: ti.f32):
        for i in range(NInstance):
            rotation_matrix = ti.math.rot_by_axis(ti.math.vec3(0, 1, 0), 0.01 * ti.math.sin(t))
            instances_transforms[i] = instances_transforms[i] @ rotation_matrix

    init_transforms_of_instances()

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(-1.82731234, 2.26492691, 2.27800684)
    camera.lookat(-1.13230401, 2.11502124, 1.57480579)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh_instance(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            transforms=instances_transforms,
        )
        canvas.scene(scene)

    for i in range(30):
        update_transform(30)
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_mesh_instances")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_mesh_instances_old():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    # Instance Attribute Information
    NInstanceRows = 100
    NInstanceCols = 100
    NInstance = NInstanceRows * NInstanceCols
    instances_transforms = ti.Matrix.field(4, 4, ti.f32, shape=(NInstance,))

    @ti.kernel
    def init_transforms_of_instances():
        identity = ti.Matrix.identity(ti.f32, 4)
        for i in range(NInstanceRows):
            for j in range(NInstanceCols):
                index = i * NInstanceCols + j
                instances_transforms[index] = identity
                translate_matrix = ti.math.translate(1.2 * j, 0, -1.2 * i)
                instances_transforms[index] = translate_matrix @ instances_transforms[index]

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    @ti.kernel
    def update_transform(t: ti.f32):
        for i in range(NInstance):
            rotation_matrix = ti.math.rot_by_axis(ti.math.vec3(0, 1, 0), 0.01 * ti.math.sin(t))
            instances_transforms[i] = instances_transforms[i] @ rotation_matrix

    init_transforms_of_instances()

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(-1.82731234, 2.26492691, 2.27800684)
    camera.lookat(-1.13230401, 2.11502124, 1.57480579)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh_instance(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            transforms=instances_transforms,
        )
        canvas.scene(scene)

    for i in range(30):
        update_transform(30)
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_mesh_instances")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_mesh_instances():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    # Instance Attribute Information
    NInstanceRows = 10
    NInstanceCols = 10
    NInstance = NInstanceRows * NInstanceCols
    instances_transforms = ti.Matrix.field(4, 4, ti.f32, shape=(NInstance,))

    @ti.kernel
    def init_transforms_of_instances():
        identity = ti.Matrix.identity(ti.f32, 4)
        for i in range(NInstanceRows):
            for j in range(NInstanceCols):
                index = i * NInstanceCols + j
                instances_transforms[index] = identity
                translate_matrix = ti.math.translate(1.2 * j, 0, -1.2 * i)
                instances_transforms[index] = translate_matrix @ instances_transforms[index]

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_transforms_of_instances()
    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(-1.82731234, 2.26492691, 2.27800684)
    camera.lookat(-1.13230401, 2.11502124, 1.57480579)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh_instance(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            transforms=instances_transforms,
            instance_count=10,
            instance_offset=2,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_mesh_instances")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_draw_part_of_mesh_instances_old():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    # Instance Attribute Information
    NInstanceRows = 10
    NInstanceCols = 10
    NInstance = NInstanceRows * NInstanceCols
    instances_transforms = ti.Matrix.field(4, 4, ti.f32, shape=(NInstance,))

    @ti.kernel
    def init_transforms_of_instances():
        identity = ti.Matrix.identity(ti.f32, 4)
        for i in range(NInstanceRows):
            for j in range(NInstanceCols):
                index = i * NInstanceCols + j
                instances_transforms[index] = identity
                translate_matrix = ti.math.translate(1.2 * j, 0, -1.2 * i)
                instances_transforms[index] = translate_matrix @ instances_transforms[index]

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_transforms_of_instances()
    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(-1.82731234, 2.26492691, 2.27800684)
    camera.lookat(-1.13230401, 2.11502124, 1.57480579)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh_instance(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            transforms=instances_transforms,
            instance_count=10,
            instance_offset=2,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_draw_part_of_mesh_instances")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_wireframe_mode():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.5, 1, -1)
    camera.lookat(1, 0.5, 0)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            show_wireframe=True,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_wireframe_mode")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=supported_archs)
def test_wireframe_mode_old():
    N = 10
    NV = (N + 1) ** 2
    NT = 2 * N**2
    NE = 2 * N * (N + 1) + N**2
    pos = ti.Vector.field(3, ti.f32, shape=NV)
    tri = ti.field(ti.i32, shape=3 * NT)
    edge = ti.Vector.field(2, ti.i32, shape=NE)

    @ti.kernel
    def init_pos():
        for i, j in ti.ndrange(N + 1, N + 1):
            idx = i * (N + 1) + j
            pos[idx] = ti.Vector([i / N, 1.0 - j / N, 0.5])

    @ti.kernel
    def init_tri():
        for i, j in ti.ndrange(N, N):
            tri_idx = 6 * (i * N + j)
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 2
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2
            else:
                tri[tri_idx + 0] = pos_idx
                tri[tri_idx + 1] = pos_idx + N + 1
                tri[tri_idx + 2] = pos_idx + 1
                tri[tri_idx + 3] = pos_idx + 1
                tri[tri_idx + 4] = pos_idx + N + 1
                tri[tri_idx + 5] = pos_idx + N + 2

    @ti.kernel
    def init_edge():
        for i, j in ti.ndrange(N + 1, N):
            edge_idx = i * N + j
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            edge_idx = start + j * N + i
            pos_idx = i * (N + 1) + j
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            edge_idx = start + i * N + j
            pos_idx = i * (N + 1) + j
            if (i + j) % 2 == 0:
                edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
            else:
                edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])

    init_pos()
    init_tri()
    init_edge()

    window = ti.ui.Window("test", (1024, 1024), vsync=True, show_window=False)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(1.5, 1, -1)
    camera.lookat(1, 0.5, 0)
    camera.fov(90)

    def render():
        scene.set_camera(camera)
        scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))

        scene.mesh(
            pos,
            tri,
            color=(39 / 255, 123 / 255, 192 / 255),
            two_sided=True,
            show_wireframe=True,
        )
        canvas.scene(scene)

    for _ in range(RENDER_REPEAT):
        render()
        window.get_image_buffer_as_numpy()

    render()
    verify_image(window.get_image_buffer_as_numpy(), "test_wireframe_mode")
    window.destroy()


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=[ti.vulkan])
def test_multi_windows():
    window = ti.ui.Window("x", (128, 128), vsync=True, show_window=False)
    window2 = ti.ui.Window("x2", (128, 128), vsync=True, show_window=False)
