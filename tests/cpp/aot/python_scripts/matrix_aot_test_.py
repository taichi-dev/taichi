import argparse
import os

import taichi as ti


def compile_matrix_aot(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def run(
        vec: ti.types.vector(3, ti.i32),
        mat: ti.types.matrix(2, 2, ti.i32),
        vec_arr: ti.types.ndarray(ndim=1, dtype=ti.i32),
        mat_arr: ti.types.ndarray(ndim=1, dtype=ti.i32),
    ):
        vec_val = vec[0] + vec[1] + vec[2]
        mat_val = mat[0, 0] + mat[0, 1] + mat[1, 0] + mat[1, 1]
        for i in ti.grouped(vec_arr):
            vec_arr[i] = vec_val
            mat_arr[i] = mat_val

    sym_vec = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "vec", dtype=ti.types.vector(3, ti.i32))

    sym_mat = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "mat", dtype=ti.types.matrix(2, 2, ti.i32))

    sym_vec_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vec_arr", dtype=ti.i32, ndim=1)

    sym_mat_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "mat_arr", dtype=ti.i32, ndim=1)

    g_builder = ti.graph.GraphBuilder()

    g_builder.dispatch(run, sym_vec, sym_mat, sym_vec_arr, sym_mat_arr)

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
        compile_matrix_aot(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_matrix_aot(arch=ti.cuda)
    elif args.arch == "vulkan":
        compile_matrix_aot(arch=ti.vulkan)
    elif args.arch == "metal":
        compile_matrix_aot(arch=ti.metal)
    elif args.arch == "opengl":
        compile_matrix_aot(arch=ti.opengl)
    else:
        assert False
