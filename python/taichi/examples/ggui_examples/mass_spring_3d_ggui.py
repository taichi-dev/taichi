import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

N = 128
cell_size = 1.0 / N
gravity = 0.5
stiffness = 1600
damping = 2
dt = 5e-4

ball_radius = 0.2
ball_center = ti.Vector.field(3, float, (1, ))

x = ti.Vector.field(3, float, (N, N))
v = ti.Vector.field(3, float, (N, N))

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, N * N)


def init_scene():
    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector([
            i * cell_size, j * cell_size / ti.sqrt(2),
            (N - j) * cell_size / ti.sqrt(2)
        ])
    ball_center[0] = ti.Vector([0.5, -0.5, -0.0])


@ti.kernel
def set_indices():
    for i, j in ti.ndrange(N, N):
        if i < N - 1 and j < N - 1:
            square_id = (i * (N - 1)) + j
            # 1st triangle of the square
            indices[square_id * 6 + 0] = i * N + j
            indices[square_id * 6 + 1] = (i + 1) * N + j
            indices[square_id * 6 + 2] = i * N + (j + 1)
            # 2nd triangle of the square
            indices[square_id * 6 + 3] = (i + 1) * N + j + 1
            indices[square_id * 6 + 4] = i * N + (j + 1)
            indices[square_id * 6 + 5] = (i + 1) * N + j


links = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]
links = [ti.Vector(v) for v in links]


@ti.kernel
def step():
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for d in ti.static(links):
            j = min(max(i + d, 0), [N - 1, N - 1])
            relative_pos = x[j] - x[i]
            current_length = relative_pos.norm()
            original_length = cell_size * float(i - j).norm()
            if original_length != 0:
                force += stiffness * relative_pos.normalized() * (
                    current_length - original_length) / original_length
        v[i] += force * dt
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        if (x[i] - ball_center[0]).norm() <= ball_radius:
            v[i] = ti.Vector([0.0, 0.0, 0.0])
        x[i] += dt * v[i]


@ti.kernel
def set_vertices():
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]


init_scene()
set_indices()

window = ti.ui.Window("Cloth", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

while window.running:
    for i in range(30):
        step()
    set_vertices()

    camera.position(0.5, -0.5, 2)
    camera.lookat(0.5, -0.5, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.mesh(vertices,
               indices=indices,
               color=(0.5, 0.5, 0.5),
               two_sided=True)
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0, 0))
    canvas.scene(scene)
    window.show()
