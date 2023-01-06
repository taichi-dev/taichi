import math
import tempfile

import pytest

import taichi as ti
from tests import test_utils


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


def test_deprecated_packed_true():
    with pytest.warns(
            DeprecationWarning,
            match=
            "Currently packed=True is the default setting and the switch will be removed in v1.4.0."
    ):
        ti.init(packed=True)


def test_deprecated_packed_false():
    with pytest.warns(
            DeprecationWarning,
            match=
            r"The automatic padding mode \(packed=False\) will no longer exist in v1.4.0. The switch will "
            "also be removed then. Make sure your code doesn't rely on it."):
        ti.init(packed=False)


@test_utils.test(arch=ti.vulkan)
def test_deprecated_rwtexture_type():
    n = 128

    with pytest.warns(
            DeprecationWarning,
            match=
            r"Specifying num_channels and channel_format is deprecated and will be removed in v1.5.0, please specify fmt instead"
    ):

        @ti.kernel
        def ker(tex: ti.types.rw_texture(num_dimensions=2,
                                         num_channels=1,
                                         channel_format=ti.f32,
                                         lod=0)):
            for i, j in ti.ndrange(n, n):
                ret = ti.cast(1, ti.f32)
                tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


# Note: will be removed in v1.5.0
@test_utils.test(arch=ti.vulkan)
def test_incomplete_info_rwtexture():
    n = 128

    with pytest.raises(
            ti.TaichiCompilationError,
            match=r"Incomplete type info for rw_texture, please specify its fmt"
    ):

        @ti.kernel
        def ker(tex: ti.types.rw_texture(num_dimensions=2,
                                         channel_format=ti.f32,
                                         lod=0)):
            for i, j in ti.ndrange(n, n):
                ret = ti.cast(1, ti.f32)
                tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))

    with pytest.raises(
            ti.TaichiCompilationError,
            match=r"Incomplete type info for rw_texture, please specify its fmt"
    ):

        @ti.kernel
        def ker(tex: ti.types.rw_texture(num_dimensions=2,
                                         num_channels=2,
                                         lod=0)):
            for i, j in ti.ndrange(n, n):
                ret = ti.cast(1, ti.f32)
                tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))

    with pytest.raises(
            ti.TaichiCompilationError,
            match=r"Incomplete type info for rw_texture, please specify its fmt"
    ):

        @ti.kernel
        def ker(tex: ti.types.rw_texture(num_dimensions=2, lod=0)):
            for i, j in ti.ndrange(n, n):
                ret = ti.cast(1, ti.f32)
                tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


def test_deprecated_source_inspect():
    with pytest.warns(DeprecationWarning,
                      match="Sourceinspect is deprecated since v1.4.0"):
        import os
        os.environ['USE_SOURCEINSPECT'] = '1'
        from taichi.lang._wrap_inspect import getsourcelines

        @ti.kernel
        def func():
            pass

        print(getsourcelines(func))


@pytest.mark.parametrize("value", [True, False])
def test_deprecated_dynamic_index(value):
    with pytest.warns(
            DeprecationWarning,
            match=
            "Dynamic index is supported by default and the switch will be removed in v1.5.0."
    ):
        ti.init(dynamic_index=value)
