import argparse
import os

import taichi as ti


def compile_bitmasked_aot(arch):
    ti.init(arch=arch)

    x = ti.field(ti.f32)
    block = ti.root.pointer(ti.i, 4)
    pixel = block.bitmasked(ti.i, 2)
    pixel.place(x)

    @ti.kernel
    def activate():
        x[2] = 1.0
        x[3] = 2.0
        x[4] = 4.0
        x[5] = 6.0

    @ti.kernel
    def deactivate():
        x[3] = x[2] + 4.0
        x[2] = x[3] + x[3]

        ti.deactivate(pixel, 4)
        ti.deactivate(pixel, 5)

    @ti.kernel
    def check_value_0():
        assert x[2] == 1.0
        assert x[3] == 2.0
        assert x[4] == 4.0
        assert x[5] == 6.0

        assert ti.is_active(pixel, 2)
        assert ti.is_active(pixel, 3)
        assert ti.is_active(pixel, 4)
        assert ti.is_active(pixel, 5)
        assert not ti.is_active(pixel, 0)
        assert not ti.is_active(pixel, 1)
        assert not ti.is_active(pixel, 6)
        assert not ti.is_active(pixel, 7)

    @ti.kernel
    def check_value_1():
        assert x[3] == 5.0
        assert x[2] == 10.0

        assert ti.is_active(pixel, 2)
        assert ti.is_active(pixel, 3)
        assert not ti.is_active(pixel, 4)
        assert not ti.is_active(pixel, 5)
        assert not ti.is_active(pixel, 0)
        assert not ti.is_active(pixel, 1)
        assert not ti.is_active(pixel, 6)
        assert not ti.is_active(pixel, 7)

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module(arch)

    m.add_kernel(activate, template_args={})
    m.add_kernel(check_value_0, template_args={})
    m.add_kernel(deactivate, template_args={})
    m.add_kernel(check_value_1, template_args={})

    m.add_field("x", x)

    m.save(dir_name, 'whatever')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()
    if args.arch == "cpu":
        compile_bitmasked_aot(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_bitmasked_aot(arch=ti.cuda)
    else:
        assert False
