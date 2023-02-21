import math
import tempfile

import pytest
from taichi._lib import core as _ti_core

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
def test_deprecate_texture_channel_format_num_channels():
    with pytest.warns(
            DeprecationWarning,
            match=
            'The channel_format and num_channels arguments are only required for RW textures since v1.5.0, you can remove them safely.'
    ):
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE,
                     'x',
                     ndim=2,
                     channel_format=ti.f32,
                     num_channels=1)


@test_utils.test()
def test_deprecate_rwtexture_channel_format_num_channels():
    with pytest.warns(
            DeprecationWarning,
            match=
            'The channel_format and num_channels arguments for texture will be deprecated in v1.5.0, use fmt instead.'
    ):
        ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE,
                     'x',
                     ndim=2,
                     channel_format=ti.f32,
                     num_channels=1)


@test_utils.test()
def test_deprecate_texture_ndim():
    with pytest.warns(
            DeprecationWarning,
            match=
            r'The shape argument for texture will be deprecated in v1.5.0, use ndim instead. \(Note that you no longer need the exact texture size.\)'
    ):
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE,
                     'x',
                     shape=(128, 128),
                     channel_format=ti.f32)


@test_utils.test()
def test_deprecate_rwtexture_ndim():
    with pytest.warns(
            DeprecationWarning,
            match=
            r'The shape argument for texture will be deprecated in v1.5.0, use ndim instead. \(Note that you no longer need the exact texture size.\)'
    ):
        ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE,
                     'x',
                     shape=(128, 128),
                     fmt=ti.Format.r32f)


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


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.cpu)
def test_deprecate_ti_ui_window():
    window = ti.ui.Window("Diff SPH", (256, 256), show_window=False)
    with pytest.warns(
            DeprecationWarning,
            match=
            r"`Window\.write_image\(\)` is deprecated, and it will be removed in Taichi v1\.6\.0\. "
    ):
        window.write_image("deprecate.png")


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.cpu)
def test_deprecate_ti_ui_make_camera():
    with pytest.warns(
            DeprecationWarning,
            match=
            r"`ti\.ui\.make_camera\(\)` is deprecated, and will be removed in Taichi v1\.6\.0\. "
    ):
        ti.ui.make_camera()


@test_utils.test()
def test_deprecation_in_taichi_init_py():
    with pytest.warns(
            DeprecationWarning,
            match=
            "ti.SOA is deprecated, and it will be removed in Taichi v1.6.0."):
        ti.SOA


@test_utils.test()
def test_deprecate_sparse_matrix_builder():
    with pytest.warns(
            DeprecationWarning,
            match=
            r"ti\.linalg\.sparse_matrix_builder is deprecated, and it will be removed in Taichi v1\.6\.0\."
    ):
        ti.linalg.sparse_matrix_builder()
