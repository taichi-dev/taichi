import argparse
import os

import taichi as ti


def compile_graph_aot(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def run0(base: int, arr: ti.types.ndarray(ndim=1, dtype=ti.i32)):
        for i in arr:
            arr[i] += base + i

    @ti.kernel
    def run1(base: int, arr: ti.types.ndarray(ndim=1, dtype=ti.types.vector(1, ti.i32))):
        for i in arr:
            arr[i] += base + i

    arr0 = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "arr0", dtype=ti.i32, ndim=1)

    arr1 = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "arr1", ti.types.vector(1, ti.i32), ndim=1)

    base0 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "base0", dtype=ti.i32)

    base1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "base2", dtype=ti.i32)

    base2 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "base1", dtype=ti.i32)

    g_builder = ti.graph.GraphBuilder()

    g_builder.dispatch(run0, base0, arr0)
    g_builder.dispatch(run0, base1, arr0)
    g_builder.dispatch(run0, base2, arr0)

    g_builder.dispatch(run1, base0, arr1)
    g_builder.dispatch(run1, base1, arr1)
    g_builder.dispatch(run1, base2, arr1)

    run_graph = g_builder.compile()

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    tmpdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    mod = ti.aot.Module()
    mod.add_graph("run_graph", run_graph)
    mod.save(tmpdir)


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
    elif args.arch == "metal":
        compile_graph_aot(arch=ti.metal)
    elif args.arch == "opengl":
        compile_graph_aot(arch=ti.opengl)
    else:
        assert False
