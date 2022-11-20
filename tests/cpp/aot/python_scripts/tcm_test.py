import argparse
import os

import taichi as ti


def main(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    @ti.kernel
    def run(arr: ti.types.ndarray()):
        for i in arr:
            arr[i] = i

    arr = ti.ndarray(int, shape=16)

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module()
    m.add_kernel(run, template_args={'arr': arr})
    m.archive(dir_name + "/module.tcm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "vulkan":
        main(arch=ti.vulkan)
    else:
        assert False
