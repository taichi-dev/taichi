from taichi.examples.patterns import taichi_logo

import taichi as ti

ti.init(arch=ti.vulkan)

res = (512, 512)
img = ti.Vector.field(4, dtype=float, shape=res)
pixels_arr = ti.Vector.ndarray(4, dtype=float, shape=res)

texture = ti.Texture(ti.f32, 1, (128, 128))


@ti.kernel
def make_texture(tex: ti.types.rw_texture(num_dimensions=2,
                                          num_channels=1,
                                          channel_format=ti.f32,
                                          lod=0)):
    for i, j in ti.ndrange(128, 128):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / 128), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def paint(t: ti.f32, pixels: ti.types.ndarray(field_dim=2),
          tex: ti.types.texture(num_dimensions=2)):
    for i, j in pixels:
        uv = ti.Vector([i / res[0], j / res[1]])
        warp_uv = uv + ti.Vector(
            [ti.cos(t + uv.x * 5.0),
             ti.sin(t + uv.y * 5.0)]) * 0.1
        c = ti.math.vec4(0.0)
        if uv.x > 0.5:
            c = tex.sample_lod(warp_uv, 0.0)
        else:
            c = tex.fetch(ti.cast(warp_uv * 128, ti.i32), 0)
        pixels[i, j] = [c.r, c.r, c.r, 1.0]


@ti.kernel
def copy_to_field(pixels: ti.types.ndarray(field_dim=2)):
    for I in ti.grouped(pixels):
        img[I] = pixels[I]


def main():
    _t = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 't', ti.f32)
    _pixels_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                               'pixels_arr',
                               ti.f32,
                               field_dim=2,
                               element_shape=(4, ))

    _rw_tex = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE,
                           'rw_tex',
                           channel_format=ti.f32,
                           shape=(128, 128),
                           num_channels=1)
    g_init_builder = ti.graph.GraphBuilder()
    g_init_builder.dispatch(make_texture, _rw_tex)
    g_init = g_init_builder.compile()

    g_init.run({'rw_tex': texture})
    _tex = ti.graph.Arg(ti.graph.ArgKind.TEXTURE,
                        'tex',
                        channel_format=ti.f32,
                        shape=(128, 128),
                        num_channels=1)
    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(paint, _t, _pixels_arr, _tex)
    g = g_builder.compile()

    aot = False
    if aot:
        tmpdir = 'shaders'
        mod = ti.aot.Module(ti.vulkan)
        mod.add_graph('g', g)
        mod.add_graph('g_init', g_init)
        mod.save(tmpdir, '')
    else:
        t = 0.0
        window = ti.ui.Window('UV', res)
        canvas = window.get_canvas()
        while window.running:
            g.run({'t': t, 'pixels_arr': pixels_arr, 'tex': texture})
            copy_to_field(pixels_arr)
            canvas.set_image(img)
            window.show()
            t += 0.03


if __name__ == '__main__':
    main()
