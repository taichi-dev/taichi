import argparse
import os

import taichi as ti


def compile_graph_aot(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def run0(rw_tex: ti.types.rw_texture(num_dimensions=2,
                                         num_channels=1,
                                         channel_format=ti.f32,
                                         lod=0)):
        for i, j in ti.ndrange(128, 128):
            value = ti.cast((j * 129 + i) % 2, ti.f32)
            rw_tex.store(ti.Vector([i, j]), ti.Vector([value, 0.0, 0.0, 0.0]))

    @ti.kernel
    def run1(tex: ti.types.texture(num_dimensions=2),
             rw_tex: ti.types.rw_texture(num_dimensions=2,
                                         num_channels=1,
                                         channel_format=ti.f32,
                                         lod=0)):
        for i, j in ti.ndrange(128, 128):
            value = tex.fetch(ti.Vector([i, j]), 0).x
            rw_tex.store(ti.Vector([i, j]),
                         ti.Vector([1.0 - value, 0.0, 0.0, 0.0]))

    @ti.kernel
    def run2(tex0: ti.types.texture(num_dimensions=2),
             tex1: ti.types.texture(num_dimensions=2),
             arr: ti.types.ndarray(field_dim=2)):
        for i, j in arr:
            value0 = tex0.fetch(ti.Vector([i, j]), 0)
            value1 = tex1.fetch(ti.Vector([i, j]), 0)
            arr[i, j] = value0.x + value1.x

    _tex0 = ti.graph.Arg(ti.graph.ArgKind.TEXTURE,
                         'tex0',
                         channel_format=ti.f32,
                         shape=(128, 128),
                         num_channels=1)
    _rw_tex0 = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE,
                            'rw_tex0',
                            channel_format=ti.f32,
                            shape=(128, 128),
                            num_channels=1)
    _tex1 = ti.graph.Arg(ti.graph.ArgKind.TEXTURE,
                         'tex1',
                         channel_format=ti.f32,
                         shape=(128, 128),
                         num_channels=1)
    _rw_tex1 = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE,
                            'rw_tex1',
                            channel_format=ti.f32,
                            shape=(128, 128),
                            num_channels=1)
    _arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                        'arr',
                        ti.f32,
                        field_dim=2,
                        element_shape=())

    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(run0, _rw_tex0)
    g_builder.dispatch(run1, _tex0, _rw_tex1)
    g_builder.dispatch(run2, _tex0, _tex1, _arr)
    run_graph = g_builder.compile()

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    tmpdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    mod = ti.aot.Module(arch)
    mod.add_graph('run_graph', run_graph)
    mod.save(tmpdir, '')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "vulkan":
        compile_graph_aot(arch=ti.vulkan)
    else:
        assert False
