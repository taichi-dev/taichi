import taichi as ti

res = (600, 400)
ti.init(arch=ti.cuda)

N = 100

vertices = ti.Vector.field(2, ti.f32, 3 * N)

window = ti.ui.Window("image", res)


@ti.kernel
def render_triangles(frame_id: int):
    for i in range(N):
        factor = ti.sin(frame_id / 200)**2
        vertices[i * 3] = ti.Vector([i, i]) / N * factor
        vertices[i * 3 + 1] = ti.Vector([i, i + 1]) / N * factor
        vertices[i * 3 + 2] = ti.Vector([i + 1, i]) / N * factor


img = ti.Vector.field(3, ti.f32, res)


@ti.kernel
def render_img(frame_id: int):
    for x, y in img:
        img[x, y][0] = x / res[0] * ti.sin(frame_id / 200)**2
        img[x, y][1] = y / res[1] * ti.cos(frame_id / 200)**2
        img[x, y][2] = 0


frame_id = 0
canvas = window.get_canvas()
print("created canvas")
while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    render_img(frame_id)
    render_triangles(frame_id)
    ti.sync()
    #print(frame_id)
    if (window.is_pressed(ti.ui.LMB)):
        print(window.get_cursor_pos())

    canvas.set_image(img)
    canvas.triangles(vertices=vertices, color=(0.7, 0.9, 0.5))
    #
    window.show()
