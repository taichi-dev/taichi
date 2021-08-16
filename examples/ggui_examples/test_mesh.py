import taichi as ti

res = (600, 400)
ti.init(arch=ti.cuda)

N = 100

a = ti.Vector.field(2, ti.f32, N)
b = ti.Vector.field(2, ti.f32, N)
c = ti.Vector.field(2, ti.f32, N)

window = ti.ui.Window("heyy", res)


@ti.kernel
def render_triangles(frame_id: int):
    for i in a:
        factor = ti.sin(frame_id / 200)**2
        a[i] = ti.Vector([i, i]) / N * factor
        b[i] = ti.Vector([i, i + 1]) / N * factor
        c[i] = ti.Vector([i + 1, i]) / N * factor


img = ti.Vector.field(3, ti.f32, res)


@ti.kernel
def render_img(frame_id: int):
    for x, y in img:
        img[x, y][0] = x / res[0] * ti.sin(frame_id / 200)**2
        img[x, y][1] = y / res[1] * ti.cos(frame_id / 200)**2
        img[x, y][2] = 0


numTriangles = 100
num_vertices = 3 * numTriangles
num_indices = num_vertices
vertices = ti.Vector.field(3, ti.f32, num_vertices)
normals = ti.Vector.field(3, ti.f32, num_vertices)
indices = ti.field(ti.i32, num_indices)


@ti.kernel
def render_mesh(frame_id: int):
    for i in range(numTriangles):
        vertices[3 * i] = ti.Vector([i / numTriangles, i / numTriangles, 1.0])
        vertices[3 * i + 1] = ti.Vector(
            [i / numTriangles, (i + 1) / numTriangles, 1.0])
        vertices[3 * i + 2] = ti.Vector([(i + 1) / numTriangles,
                                         i / numTriangles, 1.0])

        normals[3 * i] = ti.Vector([0, 0, 1])
        normals[3 * i + 1] = ti.Vector([0, 0, 1])
        normals[3 * i + 2] = ti.Vector([0, 0, 1])

        indices[3 * i] = 3 * i
        indices[3 * i + 1] = 3 * i + 1
        indices[3 * i + 2] = 3 * i + 2


frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    render_img(frame_id)
    render_triangles(frame_id)
    render_mesh(frame_id)
    ti.sync()
    #print(frame_id)
    if (window.is_pressed(ti.ui.LMB)):
        print(window.get_cursor_pos())

    camera.position(0, 0, 0)
    camera.lookat(0, 0, 1)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(5, 5, 5), color=(1, 1, 1))
    scene.mesh(vertices, normals, indices, color=(1, 1, 1))

    canvas.scene(scene)

    #
    window.show()
