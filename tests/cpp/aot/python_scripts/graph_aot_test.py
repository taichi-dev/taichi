import argparse
import os

import taichi as ti


def compile_graph_aot(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def run0(base: int, arr: ti.types.ndarray(field_dim=1, dtype=ti.i32)):
        for i in arr:
            arr[i] += base + i

    @ti.kernel
    def run1(base: int, arr: ti.types.ndarray(field_dim=1, dtype=ti.i32)):
        for i in arr:
            arr[i] += base + i

    @ti.kernel
    def run2(base: int, arr: ti.types.ndarray(field_dim=1, dtype=ti.i32)):
        for i in arr:
            arr[i] += base + i

    arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                       'arr',
                       ti.i32,
                       field_dim=1,
                       element_shape=(1, ))

    base0 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base0', ti.i32)

    base1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base2', ti.i32)

    base2 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'base1', ti.i32)

    g_builder = ti.graph.GraphBuilder()

    g_builder.dispatch(run0, base0, arr)
    g_builder.dispatch(run1, base1, arr)
    g_builder.dispatch(run2, base2, arr)

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
