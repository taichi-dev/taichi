import taichi as ti
from taichi._lib import core as _ti_core
from taichi.examples.patterns import taichi_logo

# _ti_core.wait_for_debugger()

ti.init(arch=ti.vulkan)

res = (512, 512)
pixels = ti.Vector.field(3, dtype=float, shape=res)


tex_format = ti.u8
tex = ti.Texture(tex_format, 1, (128, 128))
tex_ndarray = ti.ndarray(tex_format, shape=(128, 128))

@ti.kernel
def make_texture(arr : ti.types.ndarray()):
    for i, j in ti.ndrange(128, 128):
        ret = taichi_logo(ti.Vector([i, j]) / 128)
        ret = ti.cast(ret * 255, ti.u8)
        arr[i, j] = ret

make_texture(tex_ndarray)
tex.from_ndarray(tex_ndarray)

@ti.kernel
def paint(t : ti.f32, tex : ti.types.texture):
    for i, j in pixels:
        uv = ti.Vector([i / res[0], j / res[1]])
        warp_uv = uv + ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.1
        c = ti.math.vec4(0.0)
        if uv.x > 0.5:
          c = tex.sample_lod(warp_uv, 0.0)
        else:
          c = tex.fetch(ti.cast(warp_uv * 128, ti.i32), 0)
        pixels[i, j] = [c.r, c.r, c.r]

window = ti.ui.Window('UV', res)
canvas = window.get_canvas()

t = 0.0
while window.running:
    paint(t, tex)
    canvas.set_image(pixels)
    window.show()
    t += 0.03