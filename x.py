import taichi as ti

ti.init(arch=ti.vulkan)

window = ti.ui.Window('test', (640, 480), show_window=True)
canvas = window.get_canvas()

img = ti.Texture(ti.f32, 4, (512, 512))


@ti.kernel
def init_img(img: ti.types.rw_texture(num_dimensions=2,
                                      num_channels=2,
                                      channel_format=ti.f32,
                                      lod=0)):
    for i, j in ti.ndrange(512, 512):
        img.store(ti.Vector([i, j]),
                  ti.Vector([i, j, 0, 512], dt=ti.f32) / 512)


init_img(img)


def render():
    canvas.set_image(img)


for _ in range(10000):
    render()
    window.show()
render()

window.destroy()
