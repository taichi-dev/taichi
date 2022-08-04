import numpy as np
from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.util import taichi_scope
from taichi.types import vector
from taichi.types.primitive_types import f32, u8


class TextureSampler:
    def __init__(self, ptr_expr, num_dims) -> None:
        self.ptr_expr = ptr_expr
        self.num_dims = num_dims

    @taichi_scope
    def sample_lod(self, uv, lod):
        args_group = ()
        if self.num_dims == 1:
            args_group = (uv.x, lod)
        elif self.num_dims == 2:
            args_group = impl.make_expr_group(uv.x, uv.y, lod)
        elif self.num_dims == 3:
            args_group = impl.make_expr_group(uv.x, uv.y, uv.z, lod)
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
        args_group = ()
        if self.num_dims == 1:
            args_group = impl.make_expr_group(index.x, lod)
        elif self.num_dims == 2:
            args_group = impl.make_expr_group(index.x, index.y, lod)
        elif self.num_dims == 3:
            args_group = impl.make_expr_group(index.x, index.y, index.z, lod)
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
        self.ptr_expr = ptr_expr
        self.num_dims = num_dims

    @taichi_scope
    def load(self, index):
        args_group = ()
        if self.num_dims == 1:
            args_group = impl.make_expr_group(index.x)
        elif self.num_dims == 2:
            args_group = impl.make_expr_group(index.x, index.y)
        elif self.num_dims == 3:
            args_group = impl.make_expr_group(index.x, index.y, index.z)
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
        args_group = ()
        if self.num_dims == 1:
            args_group = impl.make_expr_group(index.x, value.r, value.g,
                                              value.b, value.a)
        elif self.num_dims == 2:
            args_group = impl.make_expr_group(index.x, index.y, value.r,
                                              value.g, value.b, value.a)
        elif self.num_dims == 3:
            args_group = impl.make_expr_group(index.x, index.y, index.z,
                                              value.r, value.g, value.b,
                                              value.a)
        impl.expr_init(
            _ti_core.make_texture_op_expr(_ti_core.TextureOpType.kStore,
                                          self.ptr_expr, args_group))


class Texture:
    """Taichi Texture class.

    Args:
        dtype (DataType): Data type of each value.
        num_channels (int): Number of channels in texture
        shape (Tuple[int]): Shape of the Texture.
    """
    def __init__(self, dtype, num_channels, arr_shape):
        self.tex = impl.get_runtime().prog.create_texture(
            dtype, num_channels, arr_shape)
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
        image = image.transpose(Image.Transpose.ROTATE_90)
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
        return Image.fromarray(res).transpose(Image.TRANSPOSE.ROTATE_270)
