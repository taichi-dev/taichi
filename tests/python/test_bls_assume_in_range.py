import taichi as ti
from .bls_test_template import bls_scatter


@ti.test(extensions=[ti.extension.bls])
def test_scattering():
    bls_scatter(N=128, ppc=10, block_size=8)


# TODO: test scatter
# TODO: debug mode behavior of assume_in_range
