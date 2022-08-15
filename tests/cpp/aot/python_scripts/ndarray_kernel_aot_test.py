import argparse
import os

import taichi as ti


def compile_ndarray_kernel_aot_test(arch):
    ti.init(arch)

    @ti.kernel
    def ker1(arr: ti.types.ndarray()):
        arr[1] = 1
        arr[2] += arr[0]

    @ti.kernel
    def ker2(arr: ti.types.ndarray(), n: ti.i32):
        arr[1] = n

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    arr = ti.ndarray(ti.i32, shape=(10, ))
    m = ti.aot.Module(arch)
    m.add_kernel(ker1, template_args={'arr': arr})
    m.add_kernel(ker2, template_args={'arr': arr})
    m.save(dir_name, 'whatever')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()
    # TODO: add test agaist cpu and cuda as well
    if args.arch == "vulkan":
        compile_ndarray_kernel_aot_test(arch=ti.vulkan)
    else:
        assert False
