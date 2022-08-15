import argparse
import os

import taichi as ti


def compile_dense_field_aot_test(arch):
    ti.init(arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    n = 10
    place = ti.field(ti.i32, shape=(n, ))

    @ti.kernel
    def simple_return() -> ti.f32:
        sum = 0.2
        return sum

    @ti.kernel
    def init():
        for index in range(n):
            place[index] = index

    @ti.kernel
    def ret() -> ti.f32:
        sum = 0.
        for index in place:
            sum += place[index]
        return sum

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module(arch)
    m.add_kernel(simple_return)
    m.add_kernel(init)
    m.add_kernel(ret)
    m.add_field("place", place)
    m.save(dir_name, 'whatever')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()
    # TODO: add test agaist cpu and cuda as well
    if args.arch == "vulkan":
        compile_dense_field_aot_test(arch=ti.vulkan)
    elif args.arch == "opengl":
        compile_dense_field_aot_test(arch=ti.opengl)
    else:
        assert False
