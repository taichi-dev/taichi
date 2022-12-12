import argparse
import os

import taichi as ti


def compile_graph_aot(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                       'arr',
                       ti.i32,
                       field_dim=1,
                       element_shape=(1, ))

    base0 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base0', ti.i32)

    base1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base2', ti.i32)

    base2 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base1', ti.i32)

    g_builder = ti.graph.GraphBuilder()

    run_graph = g_builder.compile()

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    tmpdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    mod = ti.aot.Module()
    mod.add_graph('run_graph', run_graph)
    mod.save(tmpdir)
    mod.archive(tmpdir + '/module.tcm')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "cpu":
        compile_graph_aot(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_graph_aot(arch=ti.cuda)
    elif args.arch == "vulkan":
        compile_graph_aot(arch=ti.vulkan)
    elif args.arch == "opengl":
        compile_graph_aot(arch=ti.opengl)
    else:
        assert False
