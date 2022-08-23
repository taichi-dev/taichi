import argparse
import os

import taichi as ti


def compile_dynamic_aot(arch):
    ti.init(arch=arch)

    x = ti.field(ti.i32)
    block = ti.root.dense(ti.i, 5)
    pixel = block.dynamic(ti.j, 5)
    pixel.place(x)

    @ti.kernel
    def activate():
        ti.append(x.parent(), 0, 1)
        ti.append(x.parent(), 0, 3)
        ti.append(x.parent(), 0, 5)
        ti.append(x.parent(), 1, 7)
        ti.append(x.parent(), 1, 7)
        ti.append(x.parent(), 2, 9)
        ti.append(x.parent(), 3, 12)

    @ti.kernel
    def deactivate():
        ti.deactivate(x.parent(), 0)
        x[1, 0] += 2

    @ti.kernel
    def check_value_0():
        assert ti.length(x.parent(), 0) == 3
        assert ti.length(x.parent(), 1) == 2
        assert ti.length(x.parent(), 2) == 1
        assert ti.length(x.parent(), 3) == 1
        assert ti.length(x.parent(), 4) == 0

        assert x[0, 0] == 1
        assert x[0, 1] == 3
        assert x[0, 2] == 5
        assert x[1, 0] == 7
        assert x[1, 1] == 7
        assert x[2, 0] == 9
        assert x[3, 0] == 12

    @ti.kernel
    def check_value_1():
        assert ti.length(x.parent(), 0) == 0
        assert ti.length(x.parent(), 1) == 2
        assert ti.length(x.parent(), 2) == 1
        assert ti.length(x.parent(), 3) == 1
        assert ti.length(x.parent(), 4) == 0

        assert x[0, 0] == 0
        assert x[0, 1] == 0
        assert x[0, 2] == 0
        assert x[1, 0] == 9
        assert x[1, 1] == 7
        assert x[2, 0] == 9
        assert x[3, 0] == 12

    m = ti.aot.Module(arch)

    m.add_kernel(activate, template_args={})
    m.add_kernel(check_value_0, template_args={})
    m.add_kernel(deactivate, template_args={})
    m.add_kernel(check_value_1, template_args={})

    m.add_field("x", x)

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    tmpdir = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m.save(tmpdir, 'whatever')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "cpu":
        compile_dynamic_aot(arch=ti.cpu)
    elif args.arch == "cuda":
        compile_dynamic_aot(arch=ti.cuda)
    else:
        assert False
