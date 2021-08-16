import taichi as ti

res = (600, 400)
ti.init(arch=ti.cuda)
img = ti.Vector.field(3, ti.f32, res)

window = ti.ui.Window("heyy", res)


@ti.kernel
def render(frame_id: int):
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

    render(frame_id)
    ti.sync()
    #print(frame_id)
    if (window.is_pressed(ti.ui.LMB)):
        print(window.get_cursor_pos())

    ##print("rendering ",frame_id)
    canvas.set_image(img)
    ##print("done rendering ",frame_id)
    window.show()
