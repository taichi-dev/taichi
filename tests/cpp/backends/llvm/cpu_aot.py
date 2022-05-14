import os
import pathlib
import shutil

import taichi as ti

ti.init(arch=ti.x64)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', '..', 'build')


@ti.kernel
def run():
    for i in range(16):
        print(i)


run()

dir_name = os.path.join(BUILD_DIR, 'generated')
shutil.rmtree(dir_name, ignore_errors=True)
pathlib.Path(dir_name).mkdir(parents=True, exist_ok=False)

m = ti.aot.Module(ti.x64)
m.add_kernel(run)
m.save(dir_name, 'x64-aot')
