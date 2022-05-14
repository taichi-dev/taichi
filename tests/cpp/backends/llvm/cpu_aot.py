import os
import pathlib
import shutil

import taichi as ti

ti.init(arch=ti.x64)
ti.set_logging_level(ti.TRACE)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', '..', 'build')


@ti.kernel
def run(base: int, arr: ti.types.ndarray()):
    for i in arr:
        arr[i] = base + i


arr = ti.ndarray(int, shape=16)
run(42, arr)
print(arr.to_numpy())

dir_name = os.path.join(BUILD_DIR, 'generated')
shutil.rmtree(dir_name, ignore_errors=True)
pathlib.Path(dir_name).mkdir(parents=True, exist_ok=False)

m = ti.aot.Module(ti.x64)
m.add_kernel(run, template_args={'arr': arr})
m.save(dir_name, 'x64-aot')
