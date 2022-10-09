import argparse
import os

import taichi as ti


def compile_kernel_aot_test1(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def run(base: int, arr: ti.types.ndarray()):
        for i in arr:
            arr[i] = base + i

    arr = ti.ndarray(int, shape=16)

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module(arch)
    m.add_kernel(run, template_args={'arr': arr})
    m.save(dir_name, 'whatever')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "cpu":
        compile_kernel_aot_test1(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_kernel_aot_test1(arch=ti.cuda)
    elif args.arch == "vulkan":
        compile_kernel_aot_test1(arch=ti.vulkan)
    elif args.arch == "opengl":
        compile_kernel_aot_test1(arch=ti.opengl)
    else:
        assert False
