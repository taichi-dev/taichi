import math
import tempfile

import pytest
from taichi._lib import core as _ti_core

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_remove_element_shape_scalar():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match="The element_shape argument for scalar is deprecated in v1.6.0, and is removed in v1.7.0. "
        "Please remove them.",
    ):
        sym_x = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "x", dtype=ti.f32, element_shape=())


@test_utils.test()
def test_remove_element_shape_ndarray_arg():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match="The element_shape argument for ndarray is deprecated in v1.6.0, and it is removed in v1.7.0. "
        "Please use vector or matrix data type instead.",
    ):
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1, element_shape=(1,))


@test_utils.test()
def test_remove_texture_channel_format_num_channels():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match="The channel_format and num_channels arguments are no longer required for non-RW textures "
        "since v1.6.0, and they are removed in v1.7.0. Please remove them.",
    ):
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "x", ndim=2, channel_format=ti.f32, num_channels=1)


@test_utils.test()
def test_remove_rwtexture_channel_format_num_channels():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match="The channel_format and num_channels arguments for texture are deprecated in v1.6.0, "
        "and they are removed in v1.7.0. Please use fmt instead.",
    ):
        ti.graph.Arg(
            ti.graph.ArgKind.RWTEXTURE,
            "x",
            ndim=2,
            channel_format=ti.f32,
            num_channels=1,
        )


@test_utils.test()
def test_remove_texture_ndim():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match=r"The shape argument for texture is deprecated in v1.6.0, and it is removed in v1.7.0. "
        r"Please use ndim instead. \(Note that you no longer need the exact texture size.\)",
    ):
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "x", shape=(128, 128), channel_format=ti.f32)


@test_utils.test()
def test_remove_rwtexture_ndim():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match=r"The shape argument for texture is deprecated in v1.6.0, and it is removed in v1.7.0. "
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


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test()
def test_deprecate_initialization_of_scene():
    with pytest.warns(
        DeprecationWarning,
        match=r"Instantiating ti.ui.Scene directly is deprecated, use the get_scene\(\) function from a taichi.ui.Window object instead.",
    ):
        ti.ui.Scene()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_deprecate_experimental_real_func():
    with pytest.warns(
        DeprecationWarning,
        match="ti.experimental.real_func is deprecated because it is no longer experimental. "
        "Use ti.real_func instead.",
    ):

        @ti.experimental.real_func
        def foo(a: ti.i32) -> ti.i32:
            s = 0
            for i in range(100):
                if i == a + 1:
                    return s
                s = s + i
            return s

        @ti.kernel
        def bar(a: ti.i32) -> ti.i32:
            return foo(a)

        assert bar(10) == 11 * 5
        assert bar(200) == 99 * 50
