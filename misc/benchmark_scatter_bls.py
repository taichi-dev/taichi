import taichi as ti
import sys
sys.path.append('../tests/python/')

from bls_test_template import bls_scatter

ti.init(arch=ti.cuda, kernel_profiler=True)
bls_scatter(N=1024, ppc=10, block_size=16, benchmark=10)

ti.kernel_profiler_print()
