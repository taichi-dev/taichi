from utils import compile_kernel_aot

import taichi as ti

if __name__ == "__main__":
    compile_kernel_aot(arch=ti.x64)
