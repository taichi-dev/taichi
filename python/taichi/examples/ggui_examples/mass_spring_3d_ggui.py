import taichi as ti

ti.init(arch=ti.cuda)  # Alternatively, ti.init(arch=ti.cpu)

n = 128
cell_size = 1.0 / n
gravity = 0.5
stiffness = 2000
damping = 1
dt = 5e-4

ball_radius = 0.2
ball_center = ti.Vector.field(3, float, (1, ))

x = ti.Vector.field(3, float, (n, n))
v = ti.Vector.field(3, float, (n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, n * n)

def init_scene():
    for i, j in ti.ndrange(n, n):
        x[i, j] = [
            i * cell_size - 0.5, 0.7,
            j * cell_size - 0.5
        ]
    ball_center[0] = [0, 0, 0]

@ti.kernel
def set_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        square_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[square_id * 6 + 0] = i * n + j
        indices[square_id * 6 + 1] = (i + 1) * n + j
        indices[square_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[square_id * 6 + 3] = (i + 1) * n + j + 1
        indices[square_id * 6 + 4] = i * n + (j + 1)
        indices[square_id * 6 + 5] = (i + 1) * n + j


links = []

# Link to 8 neighbors
for i in range(-1, 2):
    for j in range(-1, 2):
        if (i, j) != (0, 0):
            links.append(ti.Vector([i, j]))

@ti.kernel
def step():
    for i in ti.grouped(x):
        v[i].y -= gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for d in ti.static(links):
            j = min(max(i + d, 0), [n - 1, n - 1])
            relative_pos = x[j] - x[i]
            current_length = relative_pos.norm()
            original_length = cell_size * float(i - j).norm()
            if original_length != 0:
                force += stiffness * relative_pos.normalized() * (
                    current_length - original_length) / original_length
        v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += dt * v[i]


@ti.kernel
def set_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


init_scene()
set_indices()

window = ti.ui.Window("Taichi Cloth Simulation", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

while window.running:
    for i in range(60):
        step()
    set_vertices()

    camera.position(0.0, 0.3, 2)
    camera.lookat(0.0, 0.3, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.mesh(vertices,
               indices=indices,
               color=(0.5, 0.5, 0.5),
               two_sided=True)
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0, 0))
    canvas.scene(scene)
    window.show()
