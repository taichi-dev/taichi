import taichi as ti

from .bls_test_template import bls_particle_grid


@ti.require(ti.extension.bls)
@ti.all_archs
def _test_scattering():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=True,
                      use_offset=False)


@ti.require(ti.extension.bls)
@ti.all_archs
def _test_scattering_offset():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=True,
                      use_offset=True)


@ti.require(ti.extension.bls)
@ti.all_archs
def _test_scattering_two_pointer_levels():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=True,
                      pointer_level=2,
                      use_offset=False)


@ti.require(ti.extension.bls)
@ti.all_archs
def test_gathering():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=False,
                      use_offset=False)


@ti.require(ti.extension.bls)
@ti.all_archs
def _test_gathering_offset():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=False,
                      use_offset=True)


# TODO: debug mode behavior of assume_in_range
