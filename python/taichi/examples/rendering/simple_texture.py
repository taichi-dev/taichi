import taichi as ti
from taichi._lib import core as _ti_core
from taichi.examples.patterns import taichi_logo

# _ti_core.wait_for_debugger()

ti.init(arch=ti.vulkan)

res = (512, 512)
pixels = ti.Vector.field(3, dtype=float, shape=res)


tex = ti.Texture(ti.f32, 4, (128, 128))

tex_ndarray = ti.Vector.ndarray(4, ti.f32, shape=(128, 128))

@ti.kernel
def make_texture(arr : ti.types.ndarray(element_dim=1)):
    for i, j in ti.ndrange(128 * 4, 128 * 4):
        # 4x4 super sampling:
        ret = taichi_logo(ti.Vector([i, j]) / (128 * 4)) / 16
        arr[i // 4, j // 4] += ti.Vector([ret, ret, ret, ret])

make_texture(tex_ndarray)
tex.from_ndarray(tex_ndarray)

@ti.kernel
def paint(t : ti.f32):
    for i, j in pixels:
        uv = ti.Vector([i / res[0], j / res[1]])
        uv += ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.1
        c = ti.sample_texture(tex, uv)
        pixels[i, j] = [c.r, c.g, c.b]

window = ti.ui.Window('UV', res)
canvas = window.get_canvas()

# what's going on here...
# FIXME: why do we need to call `paint` once before copying to texture?
paint(0)
tex.from_ndarray(tex_ndarray)

t = 0.0
while window.running:
    paint(t)
    canvas.set_image(pixels)
    window.show()
    t += 0.03