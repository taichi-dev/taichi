from utils import compile_kernel_aot

import taichi as ti

compile_kernel_aot(arch=ti.x64)
