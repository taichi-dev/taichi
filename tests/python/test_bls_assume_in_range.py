import taichi as ti
from .bls_test_template import bls_particle_grid


@ti.require(ti.extension.bls)
@ti.all_archs
def test_scattering():
    bls_particle_grid(N=128, ppc=10, block_size=8)


# TODO: test scatter
# TODO: debug mode behavior of assume_in_range
