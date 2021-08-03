import sys

import taichi as ti

sys.path.append('../tests/python/')

from bls_test_template import bls_test_template

ti.init(arch=ti.gpu,
        print_ir=True,
        kernel_profiler=True,
        demote_dense_struct_fors=False)

stencil = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
bls_test_template(2,
                  4096,
                  bs=16,
                  stencil=stencil,
                  scatter=False,
                  benchmark=True,
                  dense=True)
