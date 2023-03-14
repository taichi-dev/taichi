import argparse
import os

import taichi as ti


def main(arch):
    ti.init(arch=arch)

    if ti.lang.impl.current_cfg().arch != arch:
        return

    arr = ti.ndarray(int, shape=16)

    assert "TAICHI_AOT_FOLDER_PATH" in os.environ.keys()
    dir_name = str(os.environ["TAICHI_AOT_FOLDER_PATH"])

    m = ti.aot.Module()

    tcm_path = dir_name + "/module.tcm"
    m.save(dir_name)
    m.archive(tcm_path)
    print(tcm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    args = parser.parse_args()

    if args.arch == "vulkan":
        main(arch=ti.vulkan)
    elif args.arch == "metal":
        main(arch=ti.metal)
    else:
        assert False
