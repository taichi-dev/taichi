import pytest
from tests import test_utils
from types import SimpleNamespace

import numpy as np
from .basis_functions_ref import(
    lagmatrix as np_lagmatrix,
    hermmatrix as np_hermmatrix,
    chebmatrix as np_chebmatrix,
    legmatrix as np_legmatrix,
    polymatrix as np_polymatrix,
    fourmatrix as np_fourmatrix,
)
import taichi as ti
from taichi.math.polynomial import (
    lagmatrix as ti_lagmatrix,
    hermmatrix as ti_hermmatrix,
    chebmatrix as ti_chebmatrix,
    legmatrix as ti_legmatrix,
    polymatrix as ti_polymatrix,
    fourmatrix as ti_fourmatrix,
)

np_matrices_eval = SimpleNamespace(
    laguerre=np_lagmatrix,
    hermite=np_hermmatrix,
    legendre=np_legmatrix,
    chebyshev=np_chebmatrix,
    monomial=np_polymatrix,
    fourier=np_fourmatrix,
)

ti_matrices_eval = SimpleNamespace(
    laguerre=ti_lagmatrix,
    hermite=ti_hermmatrix,
    legendre=ti_legmatrix,
    chebyshev=ti_chebmatrix,
    monomial=ti_polymatrix,
    fourier=ti_fourmatrix,
)




def _test_basis_matrices(dt, family, num_basis_functions, use_orth_weight):
    
    # Numpy logic to get expected values
    np_dt = np.float32 if dt == ti.f32 else np.float64

    x = np.linspace(0.0, 0.999, 37, dtype=np_dt)
    np_basis_func = getattr(np_matrices_eval, family)
    expected = np_basis_func(x, x.shape[0], num_basis_functions, use_orth_weight).astype(np_dt)

    # Taichi logic to get actual values
    ti_x = ti.field(dt, shape=x.shape)
    ti_x.from_numpy(x)
    ti_matrix = ti.field(dt, shape=(x.shape[0], num_basis_functions))
    ti_basis_func = getattr(ti_matrices_eval, family)

    if use_orth_weight is not None:
        @ti.kernel
        def basis_test(ti_x: ti.template(), use_orth_weight: ti.template(), ti_matrix: ti.template()):
            ti_basis_func(ti_x, use_orth_weight, ti_matrix)

        basis_test(ti_x, use_orth_weight, ti_matrix)
    else:
        @ti.kernel
        def basis_test(ti_x: ti.template(), ti_matrix: ti.template()):
            ti_basis_func(ti_x, ti_matrix)

        basis_test(ti_x, ti_matrix)
    
    actual = ti_matrix.to_numpy()
    
    # Compare expected and actual values
    tol = 1e-5 if dt == ti.f32 else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "family,use_orth_weight",
    [
        pytest.param("laguerre", False),
        pytest.param("laguerre", True),
        pytest.param("hermite", False),
        pytest.param("hermite", True),
        pytest.param("chebyshev", False),
        pytest.param("chebyshev", True),
        pytest.param("legendre", None),
        pytest.param("monomial", None),
        pytest.param("fourier", None),
    ],
)
@pytest.mark.parametrize("num_basis_functions", [2, 4, 7, 9])
@test_utils.test(default_fp=ti.f32, fast_math=False)
def test_basis_matrices_f32(family, use_orth_weight, num_basis_functions):
    _test_basis_matrices(ti.f32, family, num_basis_functions, use_orth_weight)


@pytest.mark.parametrize(
    "family,use_orth_weight",
    [
        pytest.param("laguerre", False),
        pytest.param("laguerre", True),
        pytest.param("hermite", False),
        pytest.param("hermite", True),
        pytest.param("chebyshev", False),
        pytest.param("chebyshev", True),
        pytest.param("legendre", None),
        pytest.param("monomial", None),
        pytest.param("fourier", None),
    ],
)
@pytest.mark.parametrize("num_basis_functions", [2, 4, 7, 9])
@test_utils.test(require=ti.extension.data64, default_fp=ti.f64, fast_math=False)
def test_basis_matrices_f64(family, use_orth_weight, num_basis_functions):
    _test_basis_matrices(ti.f64, family, num_basis_functions, use_orth_weight)
