import argparse
import os

import taichi as ti


def compile_kernel_aot_test2(arch, save_compute_graph):
    ti.init(arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def ker1(arr: ti.types.ndarray()):
        arr[1] = 1
        arr[2] += arr[0]

    @ti.kernel
    def ker2(arr: ti.types.ndarray(), n: ti.i32):
        arr[1] = n

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module(arch)
    if save_compute_graph:
        sym_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                               'arr',
                               ti.i32,
                               field_dim=1)
        sym_n = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'x', ti.i32)

        graph_builder = ti.graph.GraphBuilder()
        graph_builder.dispatch(ker1, sym_arr)
        graph_builder.dispatch(ker2, sym_arr, sym_n)
        graph = graph_builder.compile()
        m.add_graph('test', graph)
    else:
        arr = ti.ndarray(ti.i32, shape=(10, ))
        m.add_kernel(ker1, template_args={'arr': arr})
        m.add_kernel(ker2, template_args={'arr': arr})
    m.save(dir_name, 'whatever')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    parser.add_argument("--cgraph", action='store_true', default=False)

    args = parser.parse_args()
    # TODO: add test agaist cpu and cuda as well
    if args.arch == "vulkan":
        compile_kernel_aot_test2(arch=ti.vulkan,
                                 save_compute_graph=args.cgraph)
    elif args.arch == "opengl":
        compile_kernel_aot_test2(arch=ti.opengl,
                                 save_compute_graph=args.cgraph)
    else:
        assert False
