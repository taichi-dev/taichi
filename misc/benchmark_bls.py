import taichi as ti
import sys
sys.path.append('../tests/python/')

from bls_test_template import bls_test_template

ti.init(arch=ti.gpu,
        print_ir=True,
        kernel_profiler=True,
        demote_dense_struct_fors=False)

# _test_bls_stencil(1, 128, bs=32, stencil=(
#     (1, ),
#     (0, ),
# ), scatter=True, benchmark=True)

# stencil = [(i, j) for i in range(3) for j in range(3)]
# bls_test_template(2, 1024, bs=16, stencil=stencil, scatter=True, benchmark=True)
stencil = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
bls_test_template(2,
                  4096,
                  bs=16,
                  stencil=stencil,
                  scatter=False,
                  benchmark=True,
                  dense=True)
'''
def test_stencil_1d():
    # y[i] = x[i - 1] + x[i]
    _test_bls_stencil(1, 128, bs=32, stencil=((-1, ), (0, )))
'''
