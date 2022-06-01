import os
import pathlib
import shutil

import taichi as ti


def aot_compile(dir_name):
    ti.init(arch=ti.x64)
    ti.set_logging_level(ti.TRACE)

    @ti.kernel
    def run(base: int, arr: ti.types.ndarray()):
        for i in arr:
            arr[i] = base + i

    arr = ti.ndarray(int, shape=16)
    run(42, arr)
    print(arr.to_numpy())

    m = ti.aot.Module(ti.x64)
    m.add_kernel(run, template_args={'arr': arr})
    m.save(dir_name, 'x64-aot')
