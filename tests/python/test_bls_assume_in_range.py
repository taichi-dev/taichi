import taichi as ti
from tests import test_utils

from .bls_test_template import bls_particle_grid


def _test_scattering():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=True,
                      use_offset=False)


def _test_scattering_offset():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=True,
                      use_offset=True)


def _test_scattering_two_pointer_levels():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=True,
                      pointer_level=2,
                      use_offset=False)


def _test_gathering():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=False,
                      use_offset=False)


def _test_gathering_offset():
    bls_particle_grid(N=128,
                      ppc=10,
                      block_size=8,
                      scatter=False,
                      use_offset=True)


@test_utils.test(require=ti.extension.bls)
def test_gathering():
    _test_gathering()


@test_utils.test(require=ti.extension.bls)
def test_gathering_offset():
    _test_gathering_offset()


@test_utils.test(require=ti.extension.bls)
def test_scattering_two_pointer_levels():
    _test_scattering_two_pointer_levels()


@test_utils.test(require=ti.extension.bls)
def test_scattering():
    _test_scattering()


@test_utils.test(require=ti.extension.bls)
def test_scattering_offset():
    _test_scattering_offset()


@test_utils.test(require=ti.extension.bls,
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_gathering_matrix_scalarize():
    _test_gathering()


@test_utils.test(require=ti.extension.bls,
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_gathering_offset_matrix_scalarize():
    _test_gathering_offset()


@test_utils.test(require=ti.extension.bls,
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_scattering_matrix_scalarize():
    _test_scattering()


@test_utils.test(require=ti.extension.bls,
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_scattering_offset_matrix_scalarize():
    _test_scattering_offset()


@test_utils.test(require=ti.extension.bls,
                 real_matrix=True,
                 real_matrix_scalarize=True)
def test_scattering_two_pointer_levels_matrix_scalarize():
    _test_scattering_two_pointer_levels()


# TODO: debug mode behavior of assume_in_range
