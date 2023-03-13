import argparse
import os

import taichi as ti


def compile_kernel_return_aot(arch):
    ti.init(arch=arch)

    s = ti.types.struct(a=ti.i32, b=ti.math.vec3)

    @ti.kernel
    def test_ret() -> s:
        return s(1, ti.math.vec3([2, 3, 4]))

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module()

    m.add_kernel(test_ret, template_args={})

    m.save(dir_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()
    if args.arch == "cpu":
        compile_kernel_return_aot(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_kernel_return_aot(arch=ti.cuda)
    else:
        assert False
