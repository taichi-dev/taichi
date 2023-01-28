import math
import tempfile

import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_deprecate_a_atomic_b():
    with pytest.warns(
            DeprecationWarning,
            match=
            r"a\.atomic_add\(b\) is deprecated, and it will be removed in Taichi v1.6.0."
    ):

        @ti.kernel
        def func():
            a = 1
            a.atomic_add(2)

        func()


@test_utils.test()
def test_deprecate_element_shape_ndarray_annotation():
    with pytest.warns(
            DeprecationWarning,
            match=
            'The element_dim and element_shape arguments for ndarray will be deprecated in v1.5.0, use matrix dtype instead.'
    ):

        @ti.kernel
        def func(x: ti.types.ndarray(element_shape=(3, ))):
            pass


@test_utils.test()
def test_deprecate_element_dim_ndarray_annotation():
    with pytest.warns(
            DeprecationWarning,
            match=
            'The element_dim and element_shape arguments for ndarray will be deprecated in v1.5.0, use matrix dtype instead.'
    ):

        @ti.kernel
        def func(x: ti.types.ndarray(element_dim=2)):
            pass


@test_utils.test()
def test_deprecate_field_dim_ndarray_annotation():
    with pytest.warns(
            DeprecationWarning,
            match=
            "The field_dim argument for ndarray will be deprecated in v1.5.0, use ndim instead."
    ):

        @ti.kernel
        def func(x: ti.types.ndarray(field_dim=(16, 16))):
            pass


@test_utils.test()
def test_deprecate_field_dim_ndarray_arg():
    with pytest.warns(
            DeprecationWarning,
            match=
            "The field_dim argument for ndarray will be deprecated in v1.5.0, use ndim instead."
    ):
        sym_x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                             'x',
                             ti.math.vec2,
                             field_dim=1)


@test_utils.test()
def test_deprecate_element_shape_ndarray_arg():
    with pytest.warns(
            DeprecationWarning,
            match=
            'The element_shape argument for ndarray will be deprecated in v1.5.0, use vector or matrix data type instead.'
    ):
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                     'x',
                     ti.f32,
                     ndim=1,
                     element_shape=(1, ))


@test_utils.test()
def test_deprecate_builtin_min_max():
    with pytest.warns(
            DeprecationWarning,
            match=
            'Calling builtin function "max" in Taichi scope is deprecated, '
            'and it will be removed in Taichi v1.6.0.'):

        @ti.kernel
        def func():
            max(1, 2)

        func()


@test_utils.test()
def test_deprecate_is_is_not():
    with pytest.warns(DeprecationWarning,
                      match='Operator "is" in Taichi scope is deprecated, '
                      'and it will be removed in Taichi v1.6.0.'):

        @ti.kernel
        def func():
            ti.static(1 is 2)

        func()


@test_utils.test()
def test_deprecate_ndrange():
    with pytest.warns(
            DeprecationWarning,
            match=
            'Ndrange for loop with number of the loop variables not equal to '
            'the dimension of the ndrange is deprecated, '
            'and it will be removed in Taichi 1.6.0. '):

        @ti.kernel
        def func():
            for i in ti.ndrange(4, 4):
                pass

        func()


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


@test_utils.test()
def test_deprecation_in_taichi_init_py():
    with pytest.warns(
            DeprecationWarning,
            match=
            "ti.SOA is deprecated, and it will be removed in Taichi v1.6.0."):
        ti.SOA
