import taichi as ti
from .bls_test_template import bls_particle_grid


@ti.require(ti.extension.bls)
@ti.all_archs
def test_scattering():
    bls_particle_grid(N=128, ppc=10, block_size=8, scatter=True)

@ti.require(ti.extension.bls)
@ti.all_archs
def test_scattering_two_pointer_levels():
    bls_particle_grid(N=128, ppc=10, block_size=8, scatter=True, pointer_level=2)

@ti.require(ti.extension.bls)
@ti.all_archs
def test_gathering():
    bls_particle_grid(N=128, ppc=10, block_size=8, scatter=True)

# TODO: debug mode behavior of assume_in_range
