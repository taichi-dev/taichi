from utils import compile_field_aot

import taichi as ti

compile_field_aot(arch=ti.cuda)
