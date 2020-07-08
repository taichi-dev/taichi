import taichi as ti
from .bls_test_template import bls_scatter

@ti.require(ti.extension.bls)
@ti.all_archs
def test_scattering():
    bls_scatter(N=128, ppc=10, block_size=8)
    
# TODO: test scatter
