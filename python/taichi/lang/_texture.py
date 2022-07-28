from taichi._lib import core as _ti_core
from taichi.lang import impl
from taichi.lang.util import taichi_scope

import taichi as ti


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
        return ti.Vector([r, g, b, a])

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
        return ti.Vector([r, g, b, a])


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
        return ti.Vector([r, g, b, a])

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

    def from_ndarray(self, ndarray):
        self.tex.from_ndarray(ndarray.arr)

    def from_field(self, field):
        self.tex.from_snode(field.snode.ptr)

    def device_allocation_ptr(self):
        return self.tex.device_allocation_ptr()
