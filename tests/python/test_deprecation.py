import math
import tempfile

import pytest
from taichi._lib import core as _ti_core

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_deprecate_element_shape_scalar():
    with pytest.warns(
        DeprecationWarning,
        match="The element_shape argument for scalar is deprecated in v1.6.0, and will be removed in v1.7.0. "
        "Please remove them.",
    ):
        sym_x = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "x", dtype=ti.f32, element_shape=())


@test_utils.test()
def test_deprecate_element_shape_ndarray_arg():
    with pytest.warns(
        DeprecationWarning,
        match="The element_shape argument for ndarray is deprecated in v1.6.0, and it will be removed in v1.7.0. "
        "Please use vector or matrix data type instead.",
    ):
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1, element_shape=(1,))


@test_utils.test()
def test_deprecate_texture_channel_format_num_channels():
    with pytest.warns(
        DeprecationWarning,
        match="The channel_format and num_channels arguments are no longer required for non-RW textures "
        "since v1.6.0, and they will be removed in v1.7.0. Please remove them.",
    ):
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "x", ndim=2, channel_format=ti.f32, num_channels=1)


@test_utils.test()
def test_deprecate_rwtexture_channel_format_num_channels():
    with pytest.warns(
        DeprecationWarning,
        match="The channel_format and num_channels arguments for texture are deprecated in v1.6.0, "
        "and they will be removed in v1.7.0. Please use fmt instead.",
    ):
        ti.graph.Arg(
            ti.graph.ArgKind.RWTEXTURE,
            "x",
            ndim=2,
            channel_format=ti.f32,
            num_channels=1,
        )


@test_utils.test()
def test_deprecate_texture_ndim():
    with pytest.warns(
        DeprecationWarning,
        match=r"The shape argument for texture is deprecated in v1.6.0, and it will be removed in v1.7.0. "
        r"Please use ndim instead. \(Note that you no longer need the exact texture size.\)",
    ):
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "x", shape=(128, 128), channel_format=ti.f32)


@test_utils.test()
def test_deprecate_rwtexture_ndim():
    with pytest.warns(
        DeprecationWarning,
        match=r"The shape argument for texture is deprecated in v1.6.0, and it will be removed in v1.7.0. "
        r"Please use ndim instead. \(Note that you no longer need the exact texture size.\)",
    ):
        ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "x", shape=(128, 128), fmt=ti.Format.r32f)


@test_utils.test()
def test_remove_is_is_not():
    with pytest.raises(ti.TaichiSyntaxError, match='Operator "is" in Taichi scope is not supported'):

        @ti.kernel
        def func():
            ti.static(1 is 2)

        func()
