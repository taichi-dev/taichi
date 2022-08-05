from taichi.examples.patterns import taichi_logo

import taichi as ti

ti.init(arch=ti.vulkan)

res = (512, 512)
pixels = ti.Vector.field(3, dtype=float, shape=res)

k = 256
texture = ti.Texture(ti.f32, 1, (k, k))


@ti.kernel
def make_texture(tex: ti.types.rw_texture(
    num_dimensions=2, num_channels=1, channel_format=ti.f32, lod=0), n: ti.i32
                 ):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def paint(t: ti.f32, tex: ti.types.texture(num_dimensions=2), n: ti.i32):
    for i, j in pixels:
        uv = ti.Vector([i / res[0], j / res[1]])
        warp_uv = uv + ti.Vector(
            [ti.cos(t + uv.x * 5.0),
             ti.sin(t + uv.y * 5.0)]) * 0.1
        c = ti.math.vec4(0.0)
        if uv.x > 0.5:
            c = tex.sample_lod(warp_uv, 0.0)
        else:
            c = tex.fetch(ti.cast(warp_uv * n, ti.i32), 0)
        pixels[i, j] = [c.r, c.r, c.r]


def main():
    window = ti.ui.Window('UV', res)
    canvas = window.get_canvas()

    t = 0.0
    while window.running:
        make_texture(texture, k)
        paint(t, texture, k)
        canvas.set_image(pixels)
        window.show()
        t += 0.03


if __name__ == '__main__':
    main()
