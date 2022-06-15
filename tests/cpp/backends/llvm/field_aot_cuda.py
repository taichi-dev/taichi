from utils import compile_field_aot

import taichi as ti

if __name__ == "__main__":
    compile_field_aot(arch=ti.cuda)
