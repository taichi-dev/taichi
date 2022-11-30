import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.enums import Format
from taichi.lang.expr import Expr, make_expr_group
from taichi.lang.matrix import Matrix
from taichi.lang.util import taichi_scope
from taichi.types import vector
from taichi.types.primitive_types import f16, f32, i8, i16, i32, u8, u16, u32


def _get_entries(mat):
    if isinstance(mat, Matrix):
        return mat.entries
    assert isinstance(mat, Expr) and mat.is_tensor()
    return impl.get_runtime().prog.current_ast_builder().expand_expr([mat.ptr])


class TextureSampler:
    def __init__(self, ptr_expr, num_dims) -> None:
        self.ptr_expr = ptr_expr
        self.num_dims = num_dims

    @taichi_scope
    def sample_lod(self, uv, lod):
        args_group = make_expr_group(*_get_entries(uv), lod)
        v = _ti_core.make_texture_op_expr(_ti_core.TextureOpType.kSampleLod,
                                          self.ptr_expr, args_group)
        r = impl.call_internal("composite_extract_0",
                               v,
                               with_runtime_context=False)
        g = impl.call_internal("composite_extract_1",
                               v,
                               with_runtime_context=False)
        b = impl.call_internal("composite_extract_2",
                               v,
                               with_runtime_context=False)
        a = impl.call_internal("composite_extract_3",
                               v,
                               with_runtime_context=False)
        return vector(4, f32)([r, g, b, a])

    @taichi_scope
    def fetch(self, index, lod):
        args_group = make_expr_group(*_get_entries(index), lod)
        v = _ti_core.make_texture_op_expr(_ti_core.TextureOpType.kFetchTexel,
                                          self.ptr_expr, args_group)
        r = impl.call_internal("composite_extract_0",
                               v,
                               with_runtime_context=False)
        g = impl.call_internal("composite_extract_1",
                               v,
                               with_runtime_context=False)
        b = impl.call_internal("composite_extract_2",
                               v,
                               with_runtime_context=False)
        a = impl.call_internal("composite_extract_3",
                               v,
                               with_runtime_context=False)
        return vector(4, f32)([r, g, b, a])


class RWTextureAccessor:
    def __init__(self, ptr_expr, num_dims) -> None:
        # taichi_python.TexturePtrExpression.
        self.ptr_expr = ptr_expr
        self.num_dims = num_dims

    @taichi_scope
    def load(self, index):
        args_group = make_expr_group(*_get_entries(index))
        v = _ti_core.make_texture_op_expr(_ti_core.TextureOpType.kLoad,
                                          self.ptr_expr, args_group)
        r = impl.call_internal("composite_extract_0",
                               v,
                               with_runtime_context=False)
        g = impl.call_internal("composite_extract_1",
                               v,
                               with_runtime_context=False)
        b = impl.call_internal("composite_extract_2",
                               v,
                               with_runtime_context=False)
        a = impl.call_internal("composite_extract_3",
                               v,
                               with_runtime_context=False)
        return vector(4, f32)([r, g, b, a])

    @taichi_scope
    def store(self, index, value):
        args_group = make_expr_group(*_get_entries(index), *_get_entries(value))
        impl.expr_init(
            _ti_core.make_texture_op_expr(_ti_core.TextureOpType.kStore,
                                          self.ptr_expr, args_group))

    @property
    @taichi_scope
    def shape(self):
        """A list containing sizes for each dimension. Note that element shape will be excluded.

        Returns:
            List[Int]: The result list.
        """
        dim = _ti_core.get_external_tensor_dim(self.ptr_expr)
        ret = [
            Expr(
                _ti_core.get_external_tensor_shape_along_axis(
                    self.ptr_expr, i)) for i in range(dim)
        ]
        return ret

    @taichi_scope
    def _loop_range(self):
        """Gets the corresponding taichi_python.Expr to serve as loop range.

        Returns:
            taichi_python.Expr: See above.
        """
        return self.ptr_expr


FORMAT2TY_CH = {
    Format.r8: (u8, 1),
    Format.r8u: (u8, 1),
    Format.r8i: (i8, 1),
    Format.rg8: (u8, 2),
    Format.rg8u: (u8, 2),
    Format.rg8i: (i8, 2),
    Format.rgba8: (u8, 4),
    Format.rgba8u: (u8, 4),
    Format.rgba8i: (i8, 4),
    Format.r16: (u16, 1),
    Format.r16u: (u16, 1),
    Format.r16i: (i16, 1),
    Format.r16f: (f16, 1),
    Format.rg16: (u16, 2),
    Format.rg16u: (u16, 2),
    Format.rg16i: (i16, 2),
    Format.rg16f: (f16, 2),
    Format.rgb16: (u16, 3),
    Format.rgb16u: (u16, 3),
    Format.rgb16i: (i16, 3),
    Format.rgb16f: (f16, 3),
    Format.rgba16: (u16, 4),
    Format.rgba16u: (u16, 4),
    Format.rgba16i: (i16, 4),
    Format.rgba16f: (f16, 4),
    Format.r32u: (u32, 1),
    Format.r32i: (i32, 1),
    Format.r32f: (f32, 1),
    Format.rg32u: (u32, 2),
    Format.rg32i: (i32, 2),
    Format.rg32f: (f32, 2),
    Format.rgb32u: (u32, 3),
    Format.rgb32i: (i32, 3),
    Format.rgb32f: (f32, 3),
    Format.rgba32u: (u32, 4),
    Format.rgba32i: (i32, 4),
    Format.rgba32f: (f32, 4),
}


class Texture:
    """Taichi Texture class.

    Args:
        dtype (DataType): Data type of each value.
        num_channels (int): Number of channels in texture
        shape (Tuple[int]): Shape of the Texture.
    """
    def __init__(self, fmt, arr_shape):
        dtype, num_channels = FORMAT2TY_CH[fmt]
        self.tex = impl.get_runtime().prog.create_texture(
            dtype, num_channels, arr_shape)
        self.fmt = fmt
        self.dtype = dtype
        self.num_channels = num_channels
        self.num_dims = len(arr_shape)
        self.shape = arr_shape

    def from_ndarray(self, ndarray):
        """Loads an ndarray to texture.

        Args:
            ndarray (ti.Ndarray): Source ndarray to load from.
        """
        self.tex.from_ndarray(ndarray.arr)

    def from_field(self, field):
        """Loads a field to texture.

        Args:
            field (ti.Field): Source field to load from.
        """
        self.tex.from_snode(field.snode.ptr)

    def _device_allocation_ptr(self):
        return self.tex.device_allocation_ptr()

    def from_image(self, image):
        """Loads a PIL image to texture. This method is only allowed a 2D texture with `ti.u8` dtype and `num_channels=4`.

        Args:
            image (PIL.Image.Image): Source PIL image to load from.

        """
        from PIL import Image  # pylint: disable=import-outside-toplevel
        assert isinstance(image, Image.Image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        assert image.size == tuple(self.shape)

        assert self.num_dims == 2
        assert self.dtype == u8
        assert self.num_channels == 4
        # Don't use transpose method since its enums are too new
        image = image.rotate(90, expand=True)
        arr = np.asarray(image)
        from taichi._kernels import \
            load_texture_from_numpy  # pylint: disable=import-outside-toplevel
        load_texture_from_numpy(self, arr)

    def to_image(self):
        """Saves a texture to a PIL image in RGB mode. This method is only allowed a 2D texture with `ti.u8` dtype and `num_channels=4`.

        Returns:
            img (PIL.Image.Image): a PIL image in RGB mode, with the same size as source texture.
        """
        assert self.num_dims == 2
        assert self.dtype == u8
        assert self.num_channels == 4
        from PIL import Image  # pylint: disable=import-outside-toplevel
        res = np.zeros(self.shape + (3, ), np.uint8)
        from taichi._kernels import \
            save_texture_to_numpy  # pylint: disable=import-outside-toplevel
        save_texture_to_numpy(self, res)
        return Image.fromarray(res).rotate(270, expand=True)
