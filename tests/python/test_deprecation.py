import math
import tempfile

import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.vulkan, ti.opengl, ti.cuda, ti.cpu])
def test_deprecated_aot_save_filename():
    density = ti.field(float, shape=(4, 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        m = ti.aot.Module()
        m.add_field('density', density)
        with pytest.warns(
                DeprecationWarning,
                match=
                r'Specifying filename is no-op and will be removed in release v1.4.0'
        ):
            m.save(tmpdir, 'filename')


@test_utils.test()
def test_deprecated_matrix_rotation2d():
    with pytest.warns(
            DeprecationWarning,
            match=
            r'`ti.Matrix.rotation2d\(\)` will be removed in release v1.4.0. Use `ti.math.rotation2d\(\)` instead.'
    ):
        a = ti.Matrix.rotation2d(math.pi / 2)


@test_utils.test()
def test_deprecate_element_shape_ndarray_annotation():
    with pytest.warns(
            DeprecationWarning,
            match=
            'The element_dim and element_shape arguments for ndarray will be deprecated in v1.4.0, use matrix dtype instead.'
    ):
        @ti.kernel
        def func(x: ti.types.ndarray(element_shape=(3,))):
            pass

@test_utils.test()
def test_deprecate_element_dim_ndarray_annotation():
    with pytest.warns(
            DeprecationWarning,
            match=
            'The element_dim and element_shape arguments for ndarray will be deprecated in v1.4.0, use matrix dtype instead.'
    ):
        @ti.kernel
        def func(x: ti.types.ndarray(element_dim=2)):
            pass