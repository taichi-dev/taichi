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
        def func(x: ti.types.ndarray(element_shape=(3, ))):
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


@test_utils.test()
def test_deprecate_field_dim_ndarray_annotation():
    with pytest.warns(
            DeprecationWarning,
            match=
            "The field_dim argument for ndarray will be deprecated in v1.4.0, use ndim instead."
    ):

        @ti.kernel
        def func(x: ti.types.ndarray(field_dim=(16, 16))):
            pass


@test_utils.test(arch=ti.metal)
def test_deprecate_metal_sparse():
    with pytest.warns(
            DeprecationWarning,
            match=
            "Pointer SNode on metal backend is deprecated, and it will be removed in v1.4.0."
    ):
        a = ti.root.pointer(ti.i, 10)
    with pytest.warns(
            DeprecationWarning,
            match=
            "Bitmasked SNode on metal backend is deprecated, and it will be removed in v1.4.0."
    ):
        b = a.bitmasked(ti.j, 10)

    with pytest.raises(
            ti.TaichiRuntimeError,
            match=
            "Dynamic SNode on metal backend is deprecated and removed in this release."
    ):
        ti.root.dynamic(ti.i, 10)
